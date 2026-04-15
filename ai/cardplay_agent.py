"""State-machine card play agent with card counting.

The agent maintains a ``CardTracker`` across tricks within a single hand,
enabling card counting, void inference, and establishment tracking.
The tracker resets automatically when a new hand begins.

Tactics implemented:
    - Opening lead selection (sequences, 4th best, singleton, partner's suit)
    - Declarer planning (winner/loser count, finesse detection)
    - Third-hand high, second-hand low, cover-an-honor
    - Ruff management and entry awareness
    - Fallback to RuleBasedPlayer-style logic for unhandled cases
"""

import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any
from enum import Enum, auto

from engine.card import Card, Suit, Rank, DECK
from engine.play import Trick
from .bridge_params import BridgeParams
from .trace import DecisionTrace


class PlayPhase(Enum):
    OPENING_LEAD = auto()
    DECLARER_LEAD = auto()
    DECLARER_FOLLOW = auto()
    DEFENDER_LEAD = auto()
    DEFENDER_FOLLOW = auto()


# ---------------------------------------------------------------------------
# Heuristic play for MC simulation — standalone, no Trick/obs dependency
# ---------------------------------------------------------------------------

def _current_best(trick_cards: dict, leader: int,
                  trump: Optional[Suit]) -> Tuple[int, Card]:
    """Return (seat, card) of the current trick winner."""
    if not trick_cards:
        return leader, None
    # Find the actual first card played (the leader or first in dict)
    if leader in trick_cards:
        best_seat = leader
        best_card = trick_cards[leader]
    else:
        best_seat = next(iter(trick_cards))
        best_card = trick_cards[best_seat]
    for seat, c in trick_cards.items():
        if c.suit == best_card.suit and c.rank > best_card.rank:
            best_seat, best_card = seat, c
        elif trump and c.suit == trump and best_card.suit != trump:
            best_seat, best_card = seat, c
    return best_seat, best_card


def _heuristic_follow(seat: int, hand: List[Card], led_suit: Suit,
                      trick_cards: dict, leader: int,
                      trump: Optional[Suit], declarer: int) -> Card:
    """Heuristic card selection for following to a trick (MC simulation).

    Implements: second-hand-low, third-hand-high, cover honor,
    win cheaply, ruff when void, discard low.
    """
    is_decl_side = (seat % 2 == declarer % 2)
    partner = (seat + 2) % 4
    following = [c for c in hand if c.suit == led_suit]

    if following:
        # Determine current winner
        winner_seat, winner_card = _current_best(trick_cards, leader, trump)
        if winner_card is None:
            return min(following, key=lambda c: c.rank)
        partner_winning = (winner_seat == partner)
        cards_played = len(trick_cards)

        # Partner winning → play low
        if partner_winning:
            return min(following, key=lambda c: c.rank)

        # Second hand (1 card played) → second-hand low, unless covering honor
        if cards_played == 1:
            led_card = trick_cards[leader]
            if led_card.rank >= Rank.JACK:
                covers = [c for c in following if c.rank > led_card.rank]
                if covers:
                    return min(covers, key=lambda c: c.rank)
            return min(following, key=lambda c: c.rank)

        # Third/fourth hand → win cheaply if possible
        winners = [c for c in following if c.rank > winner_card.rank
                   and (winner_card.suit == led_suit or
                        (trump and winner_card.suit != trump))]
        if winners:
            return min(winners, key=lambda c: c.rank)

        # Can't win → play low
        return min(following, key=lambda c: c.rank)

    # Can't follow suit — ruff or discard
    if trump:
        trumps = [c for c in hand if c.suit == trump]
        if trumps:
            winner_seat, winner_card = _current_best(trick_cards, leader, trump)
            partner_winning = (winner_card is not None and winner_seat == partner)
            if not partner_winning:
                # Ruff — use lowest trump (in simulation we don't track
                # outstanding trumps, so just ruff low)
                return min(trumps, key=lambda c: c.rank)

    # Discard lowest
    return min(hand, key=lambda c: (c.rank, c.suit))


def _heuristic_lead(seat: int, hand: List[Card],
                    trump: Optional[Suit], declarer: int) -> Card:
    """Lead selection for MC simulation.

    In MC playout there's no partner communication, so developing
    suits via low leads doesn't work. Instead, simply lead the
    highest card available (cash tricks immediately). The real
    intelligence in MC comes from the follow heuristics.

    Real-game leads (4th best, sequences, partner's suit) are
    handled by the main agent's _opening_lead/_defender_lead.
    """
    return max(hand, key=lambda c: c.rank)


# ---------------------------------------------------------------------------
# CardTracker — counts cards and infers distributions
# ---------------------------------------------------------------------------

class CardTracker:
    """Tracks played cards and infers remaining holdings.

    Enhanced with:
    - Count tracking: estimated cards per seat per suit
    - Honor placement inference: likely missing high cards
    - Vacant places suit split estimation
    """

    def __init__(self, my_hand: List[Card], dummy_hand: Optional[List[Card]],
                 my_seat: int, dummy_seat: int):
        self.my_seat = my_seat
        self.dummy_seat = dummy_seat
        self.played: Set[Card] = set()
        self.shown_void: Dict[int, Set[Suit]] = {i: set() for i in range(4)}

        # All cards in the deck
        all_cards = set(DECK)
        known = set(my_hand)
        if dummy_hand:
            known |= set(dummy_hand)
        self.unknown = all_cards - known

        # Count tracking: remaining cards per seat per suit (-1 = unknown)
        # For known hands (my_seat, dummy_seat) we know exactly.
        # For opponents, we track only total remaining and voids.
        self.opp_seats = [s for s in range(4) if s != my_seat and s != dummy_seat]
        self.opp_remaining: Dict[int, int] = {s: 13 for s in self.opp_seats}
        self.cards_played_by: Dict[int, List[Card]] = {i: [] for i in range(4)}

        # Honor placement: cards an opponent likely does NOT hold
        # (inferred when they play low following suit with a winning card available)
        self.likely_missing: Dict[int, Set[Card]] = {s: set() for s in range(4)}

    def update_trick(self, trick: Trick):
        """Record a completed trick and infer information."""
        led = trick.led_suit()
        winner_seat = trick.winner()
        winning_card = trick.cards[winner_seat]

        for seat, card in trick.cards.items():
            self.played.add(card)
            self.unknown.discard(card)
            self.cards_played_by[seat].append(card)

            if seat in self.opp_remaining:
                self.opp_remaining[seat] -= 1

            # Void inference
            if led and card.suit != led:
                self.shown_void[seat].add(led)

        # Honor placement inference: if a defender followed suit with a
        # low card, they likely don't hold high cards in that suit
        # (especially honors they could have played but chose not to)
        if led:
            for seat, card in trick.cards.items():
                if seat == self.my_seat or seat == self.dummy_seat:
                    continue
                if card.suit == led:
                    # This opponent played 'card' in the led suit
                    # They likely don't hold unknown cards ranked higher
                    # than what they played (they would have played higher
                    # to try to win, or at least signal)
                    for uc in list(self.unknown):
                        if uc.suit == led and uc.rank > card.rank:
                            self.likely_missing[seat].add(uc)

    def cards_outstanding(self, suit: Suit) -> List[Card]:
        """Cards in *suit* not in any known hand and not yet played."""
        return sorted([c for c in self.unknown if c.suit == suit],
                      key=lambda c: c.rank, reverse=True)

    def high_card_master(self, suit: Suit) -> Optional[Card]:
        """Highest outstanding card in *suit*, or None if all played."""
        outstanding = self.cards_outstanding(suit)
        return outstanding[0] if outstanding else None

    def is_established(self, card: Card, my_hand: List[Card],
                       dummy_hand: Optional[List[Card]]) -> bool:
        """True if *card* will win its trick (no higher card outstanding)."""
        outstanding = self.cards_outstanding(card.suit)
        for c in outstanding:
            if c.rank > card.rank and c not in (my_hand or []) and c not in (dummy_hand or []):
                return False
        return True

    def opponent_is_void(self, seat: int, suit: Suit) -> bool:
        """True if *seat* has shown out of *suit*."""
        return suit in self.shown_void[seat]

    def suit_split_estimate(self, suit: Suit) -> Dict[int, int]:
        """Estimate how cards in *suit* split between opponents.

        Uses vacant places: opponents with more total remaining cards
        are more likely to hold cards in the given suit.

        Returns:
            Dict mapping opponent seat -> estimated cards in suit.
        """
        outstanding = self.cards_outstanding(suit)
        n_outstanding = len(outstanding)
        if n_outstanding == 0:
            return {s: 0 for s in self.opp_seats}

        # Vacant places method
        total_vacant = sum(self.opp_remaining[s] for s in self.opp_seats)
        if total_vacant == 0:
            return {s: 0 for s in self.opp_seats}

        result = {}
        for s in self.opp_seats:
            if suit in self.shown_void[s]:
                result[s] = 0
            else:
                # Proportional to remaining hand size
                result[s] = round(n_outstanding * self.opp_remaining[s] / total_vacant)

        return result

    def should_finesse_or_drop(self, suit: Suit, missing_honor: Card,
                                total_combined: int) -> str:
        """Recommend finesse vs drop for a missing honor.

        Based on the restricted choice principle and vacant places:
        - With 9+ combined cards missing Q: play for the drop
        - With 8 combined: finesse (slightly better)
        - Honor in likely_missing for one opponent: finesse the other

        Returns:
            'drop', 'finesse_lho', 'finesse_rho', or 'either'
        """
        lho = (self.my_seat + 1) % 4
        rho = (self.my_seat + 3) % 4

        # Check likely_missing inference
        if missing_honor in self.likely_missing.get(lho, set()):
            return 'finesse_lho'  # LHO doesn't have it, finesse RHO
        if missing_honor in self.likely_missing.get(rho, set()):
            return 'finesse_rho'

        # Standard guideline: "eight ever, nine never"
        if total_combined >= 9:
            return 'drop'
        return 'either'  # finesse is slightly better with 8


# ---------------------------------------------------------------------------
# DeclarerPlan
# ---------------------------------------------------------------------------

@dataclass
class DeclarerPlan:
    """High-level plan for declarer play, recomputed each trick."""
    winners: int = 0
    target: int = 0
    shortfall: int = 0
    finesse_suits: List[Tuple[Suit, str]] = field(default_factory=list)  # (suit, 'declarer'|'dummy')
    promotion_suits: List[Suit] = field(default_factory=list)
    ruff_potential: int = 0


# ---------------------------------------------------------------------------
# StateMachineCardPlayer
# ---------------------------------------------------------------------------

class StateMachineCardPlayer:
    """Card play agent with counting and tactical play.

    Args:
        seat: Seat index (0=N, 1=E, 2=S, 3=W).
    """

    def __init__(self, seat: int, params=None):
        self.seat = seat
        self.params = params or BridgeParams()
        self.tracker = None
        self.plan = None
        self._last_trick_count = -1
        self.last_trace: Optional[DecisionTrace] = None

    # ------------------------------------------------------------------
    # tracker management
    # ------------------------------------------------------------------

    def _sync_tracker(self, obs: dict):
        """Initialize or update the card tracker."""
        completed = obs.get('completed_tricks', [])
        dummy_hand = obs.get('dummy_hand')
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1

        # Reset on new hand
        if not completed and (self.tracker is None or self._last_trick_count > 0):
            self.tracker = CardTracker(
                my_hand=obs['hand'],
                dummy_hand=dummy_hand,
                my_seat=self.seat,
                dummy_seat=dummy_seat,
            )
            self._last_trick_count = 0
            self.plan = None

        if self.tracker is None:
            return

        # Process any new completed tricks
        for i in range(self._last_trick_count, len(completed)):
            self.tracker.update_trick(completed[i])
        self._last_trick_count = len(completed)

    # ------------------------------------------------------------------
    # declarer planning
    # ------------------------------------------------------------------

    def _make_plan(self, obs: dict) -> DeclarerPlan:
        """Build a declarer plan from the current position."""
        hand = obs['hand']
        dummy = obs.get('dummy_hand', [])
        trump = obs.get('trump')
        contract = obs.get('contract')
        target = (contract.level + 6) if contract else 13
        trump_suit = trump if trump != Suit.NT else None

        winners = self._count_top_winners(hand, dummy, trump_suit)
        shortfall = max(0, target - winners)

        finesses = []
        promotions = []
        ruffs = 0

        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if suit == trump_suit:
                continue
            my_cards = [c for c in hand if c.suit == suit]
            dum_cards = [c for c in dummy if c.suit == suit]
            combined = my_cards + dum_cards

            # Finesse detection with positional awareness
            my_ranks = {c.rank for c in my_cards}
            dum_ranks = {c.rank for c in dum_cards}
            all_ranks = my_ranks | dum_ranks

            # AQ tenace: finesse K — lead from opposite hand toward AQ
            if Rank.ACE in all_ranks and Rank.QUEEN in all_ranks and Rank.KING not in all_ranks:
                if Rank.QUEEN in my_ranks:
                    finesses.append((suit, 'declarer'))  # Q in my hand, lead from dummy
                elif Rank.QUEEN in dum_ranks:
                    finesses.append((suit, 'dummy'))     # Q in dummy, lead from declarer

            # KJ tenace: finesse Q — lead toward KJ
            elif Rank.KING in all_ranks and Rank.JACK in all_ranks and Rank.QUEEN not in all_ranks:
                if Rank.KING in my_ranks:
                    finesses.append((suit, 'declarer'))
                elif Rank.KING in dum_ranks:
                    finesses.append((suit, 'dummy'))

            # K without A: finesse A — lead toward K
            elif Rank.KING in all_ranks and Rank.ACE not in all_ranks:
                if Rank.KING in my_ranks:
                    finesses.append((suit, 'declarer'))
                elif Rank.KING in dum_ranks:
                    finesses.append((suit, 'dummy'))

            # Q without K: finesse K — lead toward Q
            elif Rank.QUEEN in all_ranks and Rank.KING not in all_ranks:
                if Rank.QUEEN in my_ranks:
                    finesses.append((suit, 'declarer'))
                elif Rank.QUEEN in dum_ranks:
                    finesses.append((suit, 'dummy'))

            # Promotion: long suit with top cards, need to drive out stoppers
            if len(combined) >= 5:
                top = sorted(combined, key=lambda c: c.rank, reverse=True)
                if top[0].rank >= Rank.QUEEN:
                    promotions.append(suit)

        # Ruff potential: shortness in one hand + trumps in the other
        if trump_suit:
            dum_trumps = sum(1 for c in dummy if c.suit == trump_suit)
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit == trump_suit:
                    continue
                dum_len = sum(1 for c in dummy if c.suit == suit)
                max_ruff = self.params.max_ruff_potential
                if dum_len < max_ruff and dum_trumps > 0:
                    ruffs += min(dum_trumps, max(0, max_ruff - dum_len))
            # Can't ruff more times than dummy has trumps
            ruffs = min(ruffs, dum_trumps)

        return DeclarerPlan(
            winners=winners, target=target, shortfall=shortfall,
            finesse_suits=finesses, promotion_suits=promotions,
            ruff_potential=ruffs,
        )

    def _count_top_winners(self, hand: List[Card], dummy: List[Card],
                           trump_suit: Optional[Suit]) -> int:
        """Count sure winners from top cards in combined hands."""
        winners = 0
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            my = sorted([c for c in hand if c.suit == suit],
                        key=lambda c: c.rank, reverse=True)
            dum = sorted([c for c in dummy if c.suit == suit],
                         key=lambda c: c.rank, reverse=True)
            # Count from the top: A, K, Q... are winners if we hold them
            combined = sorted(my + dum, key=lambda c: c.rank, reverse=True)
            playable = max(len(my), len(dum))
            rank_order = [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN]
            for i, r in enumerate(rank_order):
                if i >= playable:
                    break
                if i < len(combined) and combined[i].rank == r:
                    winners += 1
                else:
                    break
        return winners

    # ------------------------------------------------------------------
    # opening lead
    # ------------------------------------------------------------------

    def _opening_lead(self, obs: dict) -> Card:
        """Select the opening lead (first trick, defending)."""
        hand = obs['hand']
        valid = obs['valid_cards']
        trump = obs.get('trump')
        calls = obs.get('calls', [])
        dealer = obs.get('dealer', 0)
        trump_suit = trump if trump != Suit.NT else None

        opp_suits = self._opp_bid_suits(calls, dealer) if getattr(
            self.params, 'avoid_leading_opp_suit', True) else set()

        # Priority 1: partner's bid suit
        partner_suit = self._partner_bid_suit(calls, dealer)
        if partner_suit:
            lead = self._lead_from_suit(valid, partner_suit, trump_suit)
            if lead:
                return lead

        # Priority 2: AK against a suit contract — lead the king (or ace if
        # king isn't present). From AKxxx+ this is a standard top-of-sequence
        # attack and sets up a ruff threat later.
        if trump_suit is not None:
            ak_lead = self._ak_lead(valid, hand, trump_suit, opp_suits)
            if ak_lead is not None:
                return ak_lead

        # Priority 3: singleton vs suit contract (cheap ruff potential),
        # preferred to broken-honor leads.
        if trump_suit is not None:
            single = self._singleton_lead(valid, hand, trump_suit, opp_suits)
            if single is not None:
                return single

        # Priority 4: top of a 3+ card honor sequence (KQJ, QJT, JT9...)
        # outside opp's bid suit.
        seq = self._find_sequence_safe(valid, trump_suit, opp_suits)
        if seq:
            return seq

        # Priority 5: vs NT — 4th from longest and strongest outside opp's
        # suit if possible.
        if trump == Suit.NT:
            best = self._longest_strongest_safe(hand, trump_suit, opp_suits)
            if best:
                lead = self._fourth_best(valid, best)
                if lead:
                    return lead

        # Priority 6: 4th best from longest (even if it's opp's suit —
        # sometimes unavoidable).
        best_suit = self._longest_strongest(hand, trump_suit)
        if best_suit:
            lead = self._fourth_best(valid, best_suit)
            if lead:
                return lead

        return max(valid, key=lambda c: c.rank)

    def _opp_bid_suits(self, calls: list, dealer: int) -> set:
        my_side = {self.seat, (self.seat + 2) % 4}
        suits = set()
        for i, c in enumerate(calls):
            if c.special:
                continue
            seat = (dealer + i) % 4
            if seat not in my_side and c.strain != Suit.NT:
                suits.add(c.strain)
        return suits

    def _ak_lead(self, valid: List[Card], hand: List[Card],
                  trump_suit: Suit, opp_suits: set) -> Optional[Card]:
        """Return the K (or A if no K) from an AK-headed suit; None if
        the holding is only AK doubleton in an opp-bid suit."""
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if suit == trump_suit or suit in opp_suits:
                continue
            in_suit = sorted([c for c in hand if c.suit == suit],
                             key=lambda c: c.rank, reverse=True)
            if (len(in_suit) >= 2 and in_suit[0].rank == Rank.ACE
                    and in_suit[1].rank == Rank.KING):
                king = in_suit[1]
                if king in valid:
                    return king
                if in_suit[0] in valid:
                    return in_suit[0]
        return None

    def _singleton_lead(self, valid: List[Card], hand: List[Card],
                         trump_suit: Suit, opp_suits: set) -> Optional[Card]:
        """Lead a singleton outside opp's bid suit and outside trumps."""
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if suit == trump_suit or suit in opp_suits:
                continue
            in_suit = [c for c in hand if c.suit == suit]
            if len(in_suit) == 1 and in_suit[0] in valid:
                return in_suit[0]
        return None

    def _find_sequence_safe(self, valid: List[Card], trump: Optional[Suit],
                             opp_suits: set) -> Optional[Card]:
        """Top of sequence, skipping any suit the opps bid."""
        by_suit: Dict[Suit, List[Card]] = {}
        for c in valid:
            if c.suit == trump or c.suit in opp_suits:
                continue
            by_suit.setdefault(c.suit, []).append(c)
        best = None
        for suit, cards in by_suit.items():
            cards = sorted(cards, key=lambda c: c.rank, reverse=True)
            if len(cards) < 2:
                continue
            for i in range(len(cards) - 1):
                if (cards[i].rank >= self.params.sequence_min_rank
                        and cards[i].rank <= Rank.KING
                        and cards[i].rank - cards[i + 1].rank == 1):
                    if best is None or cards[i].rank > best.rank:
                        best = cards[i]
                    break
        return best

    def _longest_strongest_safe(self, hand: List[Card], trump: Optional[Suit],
                                 opp_suits: set) -> Optional[Suit]:
        """Longest suit outside trumps and opp's bid suits."""
        best = None
        best_key = (-1, -1)
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if suit == trump or suit in opp_suits:
                continue
            cards = [c for c in hand if c.suit == suit]
            honors = sum(1 for c in cards if c.rank >= Rank.TEN)
            key = (len(cards), honors)
            if key > best_key:
                best_key = key
                best = suit
        return best

    def _partner_bid_suit(self, calls: list, dealer: int) -> Optional[Suit]:
        """Return the suit partner bid, if any."""
        partner = (self.seat + 2) % 4
        for i, c in enumerate(calls):
            if (dealer + i) % 4 == partner and not c.special:
                return c.strain
        return None

    def _longest_strongest(self, hand: List[Card],
                           trump: Optional[Suit]) -> Optional[Suit]:
        """Longest non-trump suit; ties broken by honor count."""
        best = None
        best_key = (-1, -1)
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if suit == trump:
                continue
            cards = [c for c in hand if c.suit == suit]
            honors = sum(1 for c in cards if c.rank >= Rank.TEN)
            key = (len(cards), honors)
            if key > best_key:
                best_key = key
                best = suit
        return best

    def _fourth_best(self, valid: List[Card], suit: Suit) -> Optional[Card]:
        """4th highest card in suit, standard NT lead."""
        cards = sorted([c for c in valid if c.suit == suit],
                       key=lambda c: c.rank, reverse=True)
        if len(cards) >= 4:
            return cards[3]
        return cards[-1] if cards else None

    def _find_sequence(self, valid: List[Card],
                       trump: Optional[Suit]) -> Optional[Card]:
        """Top of a 3+ card honor sequence (e.g., K from KQJ)."""
        by_suit: Dict[Suit, List[Card]] = {}
        for c in valid:
            if c.suit == trump:
                continue
            by_suit.setdefault(c.suit, []).append(c)

        best = None
        for suit, cards in by_suit.items():
            cards = sorted(cards, key=lambda c: c.rank, reverse=True)
            if len(cards) < 2:
                continue
            # Check for 2+ consecutive honors (KQJ, QJT, etc.)
            # Skip Ace-led sequences — AK is better led as 4th-best from length
            for i in range(len(cards) - 1):
                if (cards[i].rank >= self.params.sequence_min_rank and
                        cards[i].rank <= Rank.KING and
                        cards[i].rank - cards[i + 1].rank == 1):
                    if best is None or cards[i].rank > best.rank:
                        best = cards[i]
                    break
        return best

    def _lead_from_suit(self, valid: List[Card], suit: Suit,
                        trump: Optional[Suit]) -> Optional[Card]:
        """Standard lead from a suit: top of sequence or low."""
        cards = sorted([c for c in valid if c.suit == suit],
                       key=lambda c: c.rank, reverse=True)
        if not cards:
            return None
        # Top of touching honors
        if len(cards) >= 2 and cards[0].rank - cards[1].rank == 1 and cards[0].rank >= Rank.TEN:
            return cards[0]
        # 4th best
        if len(cards) >= 4:
            return cards[3]
        # Low from honor
        if cards[0].rank >= Rank.JACK:
            return cards[-1]
        # Top of nothing
        return cards[0]

    # ------------------------------------------------------------------
    # declarer play
    # ------------------------------------------------------------------

    def _declarer_lead(self, obs: dict) -> Card:
        """Declarer chooses which card to lead."""
        valid = obs['valid_cards']
        hand = obs['hand']
        dummy = obs.get('dummy_hand', [])
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None
        current_seat = obs.get('current_seat')
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1

        # Playing from dummy or own hand?
        playing_from_dummy = (current_seat == dummy_seat)
        active_hand = dummy if playing_from_dummy else hand
        other_hand = hand if playing_from_dummy else dummy

        # Recompute plan each trick to reflect current card distribution
        self.plan = self._make_plan(obs)

        # If we need ruffs and have shortness in dummy, lead to create ruffs
        if (trump_suit and self.plan.ruff_potential > 0
                and not playing_from_dummy):
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit == trump_suit:
                    continue
                dum_len = sum(1 for c in dummy if c.suit == suit)
                if dum_len == 0:
                    # dummy is void — lead this suit so dummy can ruff
                    suit_cards = [c for c in valid if c.suit == suit]
                    if suit_cards:
                        return min(suit_cards, key=lambda c: c.rank)

        # Finesse: lead from opposite hand toward the tenace
        if self.plan.finesse_suits:
            for fsuit, honor_hand in self.plan.finesse_suits:
                # Only lead toward the honor if we're in the opposite hand
                should_lead = (
                    (honor_hand == 'dummy' and not playing_from_dummy) or
                    (honor_hand == 'declarer' and playing_from_dummy)
                )
                if should_lead:
                    low_cards = sorted([c for c in valid if c.suit == fsuit],
                                       key=lambda c: c.rank)
                    if low_cards and low_cards[0].rank < Rank.QUEEN:
                        return low_cards[0]

        # Cash winners (established cards)
        if self.tracker:
            for c in sorted(valid, key=lambda c: c.rank, reverse=True):
                if c.suit != trump_suit and self.tracker.is_established(c, hand, dummy):
                    return c

        # Draw trumps if appropriate
        if trump_suit:
            trump_cards = sorted([c for c in valid if c.suit == trump_suit],
                                 key=lambda c: c.rank, reverse=True)
            if trump_cards and trump_cards[0].rank >= self.params.trump_draw_min:
                should_draw = True
                if self.params.trump_management_mode == 'smart':
                    # Don't draw if we still need dummy's trumps for ruffs
                    if self.plan.ruff_potential > 0 and playing_from_dummy:
                        should_draw = False
                    # Don't draw if opponents are already out of trump
                    if self.tracker:
                        opp_trumps = len(self.tracker.cards_outstanding(trump_suit))
                        # Subtract our own unplayed trumps from outstanding
                        our_trumps = sum(1 for c in hand + dummy if c.suit == trump_suit)
                        opp_trumps = max(0, opp_trumps - our_trumps)
                        if opp_trumps == 0:
                            should_draw = False
                elif self.params.trump_management_mode == 'never':
                    should_draw = False
                if should_draw:
                    return trump_cards[0]

        # Lead highest card
        return max(valid, key=lambda c: (c.rank, c.suit))

    def _declarer_follow(self, obs: dict) -> Card:
        """Declarer follows to a trick in progress."""
        valid = obs['valid_cards']
        trick = obs.get('current_trick')
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1
        partner = dummy_seat if self.seat == declarer else declarer

        if not trick:
            return valid[0]

        current_winner = self._trick_winner(trick, trump_suit)
        partner_winning = (current_winner == partner)

        led = trick.led_suit()
        following = [c for c in valid if c.suit == led]

        if following:
            if partner_winning:
                # Play low — partner is winning
                return min(following, key=lambda c: c.rank)

            # Hold-up play: in NT, when opps lead a suit and our only
            # winner is the ace, duck the first rounds to break their
            # communications.
            if (trump_suit is None
                    and self.params.use_hold_up_play):
                held_up = self._try_hold_up(
                    obs, following, trick, trump_suit,
                    declarer, dummy_seat)
                if held_up is not None:
                    return held_up

            # Try to win cheaply
            winning = [c for c in following
                       if self._beats_current(c, trick, trump_suit)]
            if winning:
                return min(winning, key=lambda c: c.rank)
            return min(following, key=lambda c: c.rank)

        # Can't follow suit — ruff or discard
        if trump_suit and not partner_winning:
            trumps = sorted([c for c in valid if c.suit == trump_suit],
                            key=lambda c: c.rank)
            if trumps:
                # Ruff high enough to prevent overruff: use cheapest trump
                # that beats any outstanding trump the opponents might play
                # after us. For simplicity, if opponents could overruff,
                # ruff with highest; otherwise ruff low.
                if self.tracker:
                    opp_trumps = self.tracker.cards_outstanding(trump_suit)
                    # Filter to just opponent trumps (exclude our own)
                    hand = obs.get('hand', [])
                    dummy = obs.get('dummy_hand', [])
                    our_trumps = {c for c in hand + dummy if c.suit == trump_suit}
                    opp_trump_ranks = [c.rank for c in opp_trumps if c not in our_trumps]
                    if opp_trump_ranks:
                        max_opp = max(opp_trump_ranks)
                        # Use cheapest trump that beats max opponent trump
                        safe = [c for c in trumps if c.rank > max_opp]
                        if safe:
                            return safe[0]  # cheapest safe ruff
                # Default: ruff with lowest
                return trumps[0]

        # Discard lowest
        return min(valid, key=lambda c: (c.rank, c.suit))

    # ------------------------------------------------------------------
    # defender play
    # ------------------------------------------------------------------

    def _defender_lead(self, obs: dict) -> Card:
        """Defender leads to a new trick (not opening lead)."""
        valid = obs['valid_cards']
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None

        # Lead established cards
        if self.tracker:
            for c in sorted(valid, key=lambda c: c.rank, reverse=True):
                if c.suit != trump_suit and self.tracker.is_established(
                        c, obs['hand'], None):
                    return c

        # Top of sequence
        seq = self._find_sequence(valid, trump_suit)
        if seq:
            return seq

        # Continue partner's suit if known
        calls = obs.get('calls', [])
        partner_suit = self._partner_bid_suit(calls, obs.get('dealer', 0))
        if partner_suit:
            ps_cards = [c for c in valid if c.suit == partner_suit]
            if ps_cards:
                return min(ps_cards, key=lambda c: c.rank)

        # Lead through declarer — avoid leading trumps
        non_trump = [c for c in valid if c.suit != trump_suit]
        if non_trump:
            return min(non_trump, key=lambda c: c.rank)

        return min(valid, key=lambda c: c.rank)

    def _defender_follow(self, obs: dict) -> Card:
        """Defender follows to a trick in progress."""
        valid = obs['valid_cards']
        trick = obs.get('current_trick')
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1
        partner = (self.seat + 2) % 4

        if not trick:
            return valid[0]

        led = trick.led_suit()
        current_winner = self._trick_winner(trick, trump_suit)
        partner_winning = (current_winner == partner)
        partner_led = (trick.leader == partner)

        following = [c for c in valid if c.suit == led]

        hand = obs.get('hand', [])

        if following:
            # Third-hand high: partner led, dummy played low
            if partner_led and not partner_winning:
                return max(following, key=lambda c: c.rank)

            # Cover an honor with an honor
            if not partner_led and len(trick.cards) == 1:
                led_card = trick.cards[trick.leader]
                if led_card.rank >= self.params.cover_honor_min:
                    covers = [c for c in following if c.rank > led_card.rank]
                    if covers:
                        return min(covers, key=lambda c: c.rank)
                # Second-hand low
                return min(following, key=lambda c: c.rank)

            # Partner winning: non-contesting, carry a signal.
            if partner_winning:
                return self._signal_card(following, led, hand, partner_led)

            # Try to win cheaply
            winning = [c for c in following
                       if self._beats_current(c, trick, trump_suit)]
            if winning:
                return min(winning, key=lambda c: c.rank)
            # Can't beat the trick — signal attitude if partner led,
            # else give a count signal when this is an early round of
            # declarer's long-suit development.
            return self._signal_card(following, led, hand, partner_led)

        # Can't follow suit — consider ruffing
        if trump_suit and not partner_winning:
            trumps = sorted([c for c in valid if c.suit == trump_suit],
                            key=lambda c: c.rank)
            if trumps:
                if self.tracker:
                    opp_trumps = self.tracker.cards_outstanding(trump_suit)
                    our_trumps = {c for c in hand if c.suit == trump_suit}
                    opp_trump_ranks = [c.rank for c in opp_trumps
                                       if c not in our_trumps]
                    if opp_trump_ranks:
                        max_opp = max(opp_trump_ranks)
                        safe = [c for c in trumps if c.rank > max_opp]
                        if safe:
                            return safe[0]
                return trumps[0]

        # Discard: signal attitude on a suit we want led.
        return self._discard_with_signal(valid, hand, trump_suit)

    def _signal_card(self, following: List[Card], led_suit: Suit,
                      hand: List[Card], partner_led: bool) -> Card:
        """Pick a card among equivalents to carry a defensive signal.

        When partner led the suit, attitude signal (high = encourage,
        low = discourage). When declarer led, count signal on the first
        round (high from even, low from odd)."""
        if len(following) == 1:
            return following[0]
        # Non-winning candidates (lowest few)
        sorted_low = sorted(following, key=lambda c: c.rank)
        # Exclude honor candidates we don't want to waste
        non_winners = [c for c in sorted_low if c.rank < Rank.TEN] or sorted_low

        if partner_led and self.params.use_attitude_signals:
            # Encourage if we hold an honor (A/K/Q) in the suit OR we have
            # a doubleton (potential ruff).
            suit_cards = [c for c in hand if c.suit == led_suit]
            has_honor = any(c.rank >= self.params.attitude_encourage_min_rank
                            for c in suit_cards)
            doubleton = len(suit_cards) == 2
            encourage = has_honor or doubleton
            if encourage and len(non_winners) >= 1:
                return non_winners[-1]  # highest spot
            return non_winners[0]       # lowest spot

        if not partner_led and self.params.use_count_signals:
            # Count signal: high from even, low from odd (original count).
            suit_cards = [c for c in hand if c.suit == led_suit]
            # Include the card about to be played in the parity.
            if len(suit_cards) % 2 == 0:
                return non_winners[-1] if non_winners else following[-1]
            return non_winners[0]

        return non_winners[0]

    def _discard_with_signal(self, valid: List[Card], hand: List[Card],
                              trump_suit: Optional[Suit]) -> Card:
        """Discard a low card from our weakest non-trump suit. Choose the
        discard suit to AVOID the one we'd most want led (standard
        negative-attitude discard)."""
        by_suit: Dict[Suit, List[Card]] = {}
        for c in valid:
            if c.suit == trump_suit:
                continue
            by_suit.setdefault(c.suit, []).append(c)
        if not by_suit:
            # Only trumps available — dump smallest
            return min(valid, key=lambda c: (c.rank, c.suit))

        def suit_value(suit: Suit) -> int:
            cards = [c for c in hand if c.suit == suit]
            return sum(max(0, c.rank - Rank.TEN + 1) for c in cards)

        # Find the suit with least high-card value — throw a small from there.
        weakest = min(by_suit.keys(), key=suit_value)
        return min(by_suit[weakest], key=lambda c: c.rank)

    # ------------------------------------------------------------------
    # trick evaluation helpers
    # ------------------------------------------------------------------

    def _trick_winner(self, trick: Trick,
                      trump: Optional[Suit]) -> Optional[int]:
        """Determine who is currently winning the trick."""
        if not trick.cards:
            return None
        led = trick.led_suit()
        best_p = trick.leader
        best_c = trick.cards[trick.leader]
        for p, c in trick.cards.items():
            if c.suit == best_c.suit and c.rank > best_c.rank:
                # Same suit (including trump vs trump): higher rank wins
                best_c, best_p = c, p
            elif trump and c.suit == trump and best_c.suit != trump:
                # First trump beats any non-trump
                best_c, best_p = c, p
        return best_p

    def _try_hold_up(self, obs: dict, following: List[Card],
                      trick: Trick, trump_suit: Optional[Suit],
                      declarer: int, dummy_seat: int) -> Optional[Card]:
        """Duck a round in NT when the only winner in the led suit is our
        ace and our side's combined holding is short enough that ducking
        exhausts one opponent.

        Returns the card to duck with, or None to fall through to normal
        'try to win cheaply' logic.
        """
        if trick is None:
            return None
        led = trick.led_suit()
        leader = trick.leader
        our_side = {declarer, dummy_seat}
        if leader in our_side:
            return None  # our side led — not a hold-up scenario

        # Our side's combined length in the led suit
        hand = obs.get('hand', []) or []
        dummy = obs.get('dummy_hand', []) or []
        declarer_len = sum(1 for c in hand if c.suit == led)
        dummy_len = sum(1 for c in dummy if c.suit == led)
        combined = declarer_len + dummy_len
        if combined > self.params.hold_up_max_combined:
            return None

        # How many rounds of this suit have the defenders already cashed?
        completed = obs.get('completed_tricks', []) or []
        rounds = sum(1 for t in completed
                     if t.cards and t.led_suit() == led)
        if rounds >= self.params.hold_up_max_rounds:
            return None

        # Duck only if the ace is our sole winner in this suit.
        winning_now = [c for c in following
                       if self._beats_current(c, trick, trump_suit)]
        if not winning_now:
            return None
        non_ace_winning = [c for c in winning_now if c.rank < Rank.ACE]
        if non_ace_winning:
            return None  # we can win without using the ace — just take it

        # Must have at least one non-ace card to duck with.
        non_ace = [c for c in following if c.rank < Rank.ACE]
        if not non_ace:
            return None

        return min(non_ace, key=lambda c: c.rank)

    def _beats_current(self, card: Card, trick: Trick,
                       trump: Optional[Suit]) -> bool:
        """True if *card* would beat the current trick winner."""
        if not trick.cards:
            return True
        led = trick.led_suit()
        best = None
        for c in trick.cards.values():
            if c.suit == led:
                if best is None or c.rank > best.rank:
                    best = c
        trump_best = None
        for c in trick.cards.values():
            if trump and c.suit == trump:
                if trump_best is None or c.rank > trump_best.rank:
                    trump_best = c

        # Card is in led suit
        if card.suit == led:
            if trump_best:
                return False  # can't beat trump by following suit
            return best is not None and card.rank > best.rank

        # Card is trump
        if trump and card.suit == trump:
            if trump_best:
                return card.rank > trump_best.rank
            return True  # first trump beats led suit

        return False

    # ------------------------------------------------------------------
    # Monte Carlo trick estimation
    # ------------------------------------------------------------------

    def _monte_carlo_play(self, obs: dict) -> Optional[Card]:
        """Sample random opponent hands and pick the card that wins most tricks.

        Returns the best card, or None if MC is disabled or can't run.
        """
        if not self.params.use_monte_carlo:
            return None

        valid = obs['valid_cards']
        if len(valid) <= 1:
            return None

        hand = obs['hand']
        dummy_hand = obs.get('dummy_hand') or []
        completed = obs.get('completed_tricks', [])
        current_trick = obs.get('current_trick')
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None
        declarer = obs.get('declarer', 0)
        current_seat = obs.get('current_seat', self.seat)
        is_declarer_side = (self.seat % 2 == declarer % 2)

        # Determine known and unknown cards
        known = set(hand) | set(dummy_hand)
        played = set()
        for t in completed:
            played |= set(t.cards.values())
        if current_trick:
            played |= set(current_trick.cards.values())

        unknown = [c for c in DECK if c not in known and c not in played]
        if len(unknown) < 2:
            return None  # too few unknowns to sample

        # Identify the two opponent seats
        opp_seats = [s for s in range(4) if s % 2 != self.seat % 2]

        # Estimate remaining hand sizes per opponent
        cards_per_seat = {}
        total_cards_played_by = {s: 0 for s in range(4)}
        for t in completed:
            for s in t.cards:
                total_cards_played_by[s] += 1
        if current_trick:
            for s in current_trick.cards:
                total_cards_played_by[s] += 1

        for s in range(4):
            cards_per_seat[s] = 13 - total_cards_played_by[s]

        opp_remaining = {s: cards_per_seat[s] for s in opp_seats}

        # Pre-compute void sets for fast rejection during dealing
        void_suits = {}
        if self.tracker:
            for s in opp_seats:
                void_suits[s] = self.tracker.shown_void.get(s, set())
        else:
            for s in opp_seats:
                void_suits[s] = set()

        dummy_seat = (declarer + 2) % 4

        # Score each candidate card with early termination
        N = self.params.monte_carlo_samples
        scores = {c: 0 for c in valid}

        for card in valid:
            for _ in range(N):
                tricks = self._simulate_one_fast(
                    card, unknown, opp_seats, opp_remaining, void_suits,
                    hand, dummy_hand, current_trick,
                    trump_suit, declarer, dummy_seat, current_seat
                )
                scores[card] += tricks

        return max(valid, key=lambda c: scores[c])

    def _simulate_one_fast(self, chosen_card, unknown, opp_seats, opp_remaining,
                           void_suits, hand, dummy_hand, current_trick,
                           trump_suit, declarer, dummy_seat, current_seat):
        """Fast single simulation: deal cards, play out, count tricks."""
        # Shuffle unknown cards (in-place for speed, restored by caller context)
        shuffled = list(unknown)
        random.shuffle(shuffled)

        # Fast dealing: single pass through shuffled cards
        opp_a, opp_b = opp_seats[0], opp_seats[1]
        need_a = opp_remaining.get(opp_a, 0)
        need_b = opp_remaining.get(opp_b, 0)
        voids_a = void_suits[opp_a]
        voids_b = void_suits[opp_b]
        hand_a = []
        hand_b = []
        overflow = []

        for c in shuffled:
            if len(hand_a) < need_a and c.suit not in voids_a:
                hand_a.append(c)
            elif len(hand_b) < need_b and c.suit not in voids_b:
                hand_b.append(c)
            else:
                overflow.append(c)

        # Assign overflow (cards rejected by voids)
        for c in overflow:
            if len(hand_a) < need_a:
                hand_a.append(c)
            elif len(hand_b) < need_b:
                hand_b.append(c)

        # Build hand map
        all_hands = {opp_a: hand_a, opp_b: hand_b}
        if current_seat == dummy_seat:
            all_hands[declarer] = list(hand)
            all_hands[dummy_seat] = [c for c in dummy_hand if c != chosen_card]
        else:
            all_hands[declarer] = [c for c in hand if c != chosen_card]
            all_hands[dummy_seat] = list(dummy_hand)

        return self._greedy_playout(
            chosen_card, current_trick, all_hands,
            trump_suit, declarer, dummy_seat, current_seat
        )

    def _greedy_playout(self, first_card, current_trick, hands,
                        trump_suit, declarer, dummy_seat, first_seat):
        """Complete the current trick and play remaining tricks using heuristics.

        Uses _heuristic_follow and _heuristic_lead instead of greedy
        "play highest" to produce realistic trick estimates.

        Returns tricks won by declarer's side.
        """
        dec_tricks = 0

        # Complete the current trick
        if current_trick and current_trick.cards:
            trick_cards = dict(current_trick.cards)
            trick_cards[first_seat] = first_card
            leader = current_trick.leader
            led_suit = current_trick.led_suit()
            order = [(leader + i) % 4 for i in range(4)]
            for seat in order:
                if seat in trick_cards:
                    continue
                h = hands.get(seat, [])
                if not h:
                    continue
                pick = _heuristic_follow(seat, h, led_suit, trick_cards,
                                         leader, trump_suit, declarer)
                trick_cards[seat] = pick
                if pick in hands[seat]:
                    hands[seat].remove(pick)

            winner = self._greedy_trick_winner(trick_cards, leader, trump_suit)
            if winner % 2 == declarer % 2:
                dec_tricks += 1
            next_leader = winner
        else:
            # We're leading — the chosen card starts a new trick
            if first_seat in hands and first_card in hands[first_seat]:
                hands[first_seat].remove(first_card)
            trick_cards = {first_seat: first_card}
            leader = first_seat
            led_suit = first_card.suit
            order = [(leader + i) % 4 for i in range(4)]
            for seat in order[1:]:
                h = hands.get(seat, [])
                if not h:
                    continue
                pick = _heuristic_follow(seat, h, led_suit, trick_cards,
                                         leader, trump_suit, declarer)
                trick_cards[seat] = pick
                hands[seat].remove(pick)

            winner = self._greedy_trick_winner(trick_cards, leader, trump_suit)
            if winner % 2 == declarer % 2:
                dec_tricks += 1
            next_leader = winner

        # Play remaining tricks with heuristic play
        for _ in range(12):
            if not any(hands.get(s) for s in range(4)):
                break
            trick_cards = {}
            order = [(next_leader + i) % 4 for i in range(4)]
            led_suit = None
            for seat in order:
                h = hands.get(seat, [])
                if not h:
                    continue
                if led_suit is None:
                    # Leader uses heuristic lead selection
                    pick = _heuristic_lead(seat, h, trump_suit, declarer)
                    led_suit = pick.suit
                else:
                    pick = _heuristic_follow(seat, h, led_suit, trick_cards,
                                             order[0], trump_suit, declarer)
                trick_cards[seat] = pick
                hands[seat].remove(pick)

            if len(trick_cards) < 2:
                break
            winner = self._greedy_trick_winner(trick_cards, next_leader, trump_suit)
            if winner % 2 == declarer % 2:
                dec_tricks += 1
            next_leader = winner

        return dec_tricks

    def _greedy_trick_winner(self, cards: dict, leader: int,
                             trump: Optional[Suit]) -> int:
        """Determine trick winner from a dict of seat→Card."""
        if not cards:
            return leader
        led_card = cards.get(leader)
        if led_card is None:
            return leader
        led_suit = led_card.suit
        best_p = leader
        best_c = led_card
        for p, c in cards.items():
            if c.suit == best_c.suit and c.rank > best_c.rank:
                best_c, best_p = c, p
            elif trump and c.suit == trump and best_c.suit != trump:
                best_c, best_p = c, p
        return best_p

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def play_card(self, obs: dict) -> Card:
        """Select a card to play.

        Args:
            obs: Observation dict from GameState.observation().

        Returns:
            A legal Card from obs['valid_cards'].
        """
        valid = obs['valid_cards']
        if len(valid) == 1:
            return valid[0]

        self._sync_tracker(obs)

        # Try Monte Carlo if enabled
        mc_result = self._monte_carlo_play(obs)
        if mc_result is not None and mc_result in valid:
            return mc_result

        trick = obs.get('current_trick')
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1
        is_declarer_side = (self.seat % 2 == (declarer or 0) % 2)
        on_lead = not trick or not trick.cards
        completed = obs.get('completed_tricks', [])

        if on_lead:
            if not is_declarer_side and len(completed) == 0:
                result = self._opening_lead(obs)
                phase_name = 'OPENING_LEAD'
            elif is_declarer_side:
                result = self._declarer_lead(obs)
                phase_name = 'DECLARER_LEAD'
            else:
                result = self._defender_lead(obs)
                phase_name = 'DEFENDER_LEAD'
        elif is_declarer_side:
            result = self._declarer_follow(obs)
            phase_name = 'DECLARER_FOLLOW'
        else:
            result = self._defender_follow(obs)
            phase_name = 'DEFENDER_FOLLOW'

        # Emit trace if enabled
        if self.params.trace_enabled:
            self.last_trace = DecisionTrace(
                action_type='play',
                seat=self.seat,
                phase=phase_name,
                chosen=str(result),
                reason=phase_name.lower(),
                candidates=[str(c) for c in valid],
                details={
                    'trick_num': len(completed) + 1,
                    'is_declarer_side': is_declarer_side,
                    'on_lead': on_lead,
                },
            )

        return result
