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
        """Cards in *suit* that are unplayed and not in known hands."""
        return sorted([c for c in self.unknown if c.suit == suit and c not in self.played],
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
            declarer = obs.get('declarer')
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit == trump_suit:
                    continue
                dum_len = sum(1 for c in dummy if c.suit == suit)
                dum_trumps = sum(1 for c in dummy if c.suit == trump_suit)
                max_ruff = self.params.max_ruff_potential
                if dum_len < max_ruff and dum_trumps > 0:
                    ruffs += min(dum_trumps, max(0, max_ruff - dum_len))

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
        trump_suit = trump if trump != Suit.NT else None

        # Priority 1: partner's bid suit
        partner_suit = self._partner_bid_suit(calls, obs.get('dealer', 0))
        if partner_suit:
            lead = self._lead_from_suit(valid, partner_suit, trump_suit)
            if lead:
                return lead

        # Priority 2: top of honor sequence (works for both NT and suit)
        seq = self._find_sequence(valid, trump_suit)
        if seq:
            return seq

        # Priority 3: vs NT — 4th from longest and strongest
        if trump == Suit.NT:
            best_suit = self._longest_strongest(hand, trump_suit)
            if best_suit:
                lead = self._fourth_best(valid, best_suit)
                if lead:
                    return lead

        # Priority 4: vs suit contract
        if trump_suit:
            # Singleton lead (hoping for ruff)
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit == trump_suit:
                    continue
                cards_in_suit = [c for c in hand if c.suit == suit]
                if len(cards_in_suit) == 1 and cards_in_suit[0] in valid:
                    return cards_in_suit[0]

        # AK lead (vs suit)
        if trump_suit:
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit == trump_suit:
                    continue
                suit_cards = sorted([c for c in hand if c.suit == suit],
                                    key=lambda c: c.rank, reverse=True)
                if (len(suit_cards) >= 2 and suit_cards[0].rank == Rank.ACE
                        and suit_cards[1].rank == Rank.KING):
                    if suit_cards[0] in valid:
                        return suit_cards[0]

        # 4th best from longest
        best_suit = self._longest_strongest(hand, trump_suit)
        if best_suit:
            lead = self._fourth_best(valid, best_suit)
            if lead:
                return lead

        # Fallback: highest card
        return max(valid, key=lambda c: c.rank)

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
            # Try to win cheaply
            winning = [c for c in following
                       if self._beats_current(c, trick, trump_suit)]
            if winning:
                return min(winning, key=lambda c: c.rank)
            return min(following, key=lambda c: c.rank)

        # Can't follow suit
        if trump_suit and not partner_winning:
            trumps = [c for c in valid if c.suit == trump_suit]
            if trumps:
                return min(trumps, key=lambda c: c.rank)

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

        if following:
            # Third-hand high: partner led, dummy played low
            if partner_led and not partner_winning:
                return max(following, key=lambda c: c.rank)

            # Cover an honor with an honor
            if not partner_led and len(trick.cards) == 1:
                # Second hand
                led_card = trick.cards[trick.leader]
                if led_card.rank >= self.params.cover_honor_min:
                    covers = [c for c in following if c.rank > led_card.rank]
                    if covers:
                        return min(covers, key=lambda c: c.rank)
                # Second-hand low
                return min(following, key=lambda c: c.rank)

            if partner_winning:
                return min(following, key=lambda c: c.rank)

            # Try to win cheaply
            winning = [c for c in following
                       if self._beats_current(c, trick, trump_suit)]
            if winning:
                return min(winning, key=lambda c: c.rank)
            return min(following, key=lambda c: c.rank)

        # Can't follow suit — consider ruffing
        if trump_suit and not partner_winning:
            trumps = [c for c in valid if c.suit == trump_suit]
            if trumps:
                return min(trumps, key=lambda c: c.rank)

        # Discard: throw lowest from weakest suit
        return min(valid, key=lambda c: (c.rank, c.suit))

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
                best_c, best_p = c, p
            elif trump and c.suit == trump and best_c.suit != trump:
                best_c, best_p = c, p
        return best_p

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

        # Score each candidate card
        N = self.params.monte_carlo_samples
        scores = {}

        for card in valid:
            total_tricks = 0
            for _ in range(N):
                tricks = self._simulate_one(
                    card, unknown, opp_seats, opp_remaining,
                    hand, dummy_hand, completed, current_trick,
                    trump_suit, declarer, current_seat, is_declarer_side
                )
                total_tricks += tricks
            scores[card] = total_tricks

        return max(valid, key=lambda c: scores[c])

    def _simulate_one(self, chosen_card, unknown, opp_seats, opp_remaining,
                      hand, dummy_hand, completed, current_trick,
                      trump_suit, declarer, current_seat, is_declarer_side):
        """Simulate one random deal and count tricks for declarer side."""
        # Assign unknown cards to opponents respecting voids
        shuffled = list(unknown)
        random.shuffle(shuffled)

        opp_hands = {s: [] for s in opp_seats}
        # Respect known voids
        void_cards = {s: [] for s in opp_seats}
        non_void = list(shuffled)

        if self.tracker:
            for s in opp_seats:
                for c in shuffled:
                    if c.suit in self.tracker.shown_void.get(s, set()):
                        void_cards[s].append(c)

        # Remove void-violating cards, assign rest proportionally
        available = list(shuffled)
        for s in opp_seats:
            need = opp_remaining.get(s, 0)
            assigned = 0
            for c in available[:]:
                if assigned >= need:
                    break
                # Skip if this opponent is void in this suit
                if self.tracker and c.suit in self.tracker.shown_void.get(s, set()):
                    continue
                opp_hands[s].append(c)
                available.remove(c)
                assigned += 1

        # Assign remaining cards to whoever still needs them
        for s in opp_seats:
            need = opp_remaining.get(s, 0) - len(opp_hands[s])
            while need > 0 and available:
                opp_hands[s].append(available.pop(0))
                need -= 1

        # Build full hand map
        dummy_seat = (declarer + 2) % 4
        all_hands = dict(opp_hands)
        # Our side's hands: remove cards already played in current trick
        my_remaining = [c for c in hand if c != chosen_card]
        dum_remaining = list(dummy_hand)

        # Figure out which seat we're playing from
        if current_seat == dummy_seat:
            dum_remaining = [c for c in dum_remaining if c != chosen_card]
        else:
            my_remaining = [c for c in my_remaining if c != chosen_card]

        all_hands[self.seat] = my_remaining
        partner_seat = (self.seat + 2) % 4
        if partner_seat == dummy_seat:
            all_hands[partner_seat] = dum_remaining
        elif partner_seat == declarer:
            all_hands[partner_seat] = dum_remaining if self.seat != declarer else my_remaining

        # Actually, let's simplify: just assign known hands
        all_hands[declarer] = [c for c in hand if c != chosen_card] if self.seat == declarer else list(hand)
        all_hands[dummy_seat] = list(dummy_hand)
        if current_seat == dummy_seat:
            all_hands[dummy_seat] = [c for c in dummy_hand if c != chosen_card]
        elif current_seat != dummy_seat and self.seat == declarer:
            all_hands[declarer] = [c for c in hand if c != chosen_card]

        # Simulate remaining tricks with greedy play
        return self._greedy_playout(
            chosen_card, current_trick, all_hands,
            trump_suit, declarer, dummy_seat, current_seat
        )

    def _greedy_playout(self, first_card, current_trick, hands,
                        trump_suit, declarer, dummy_seat, first_seat):
        """Complete the current trick and play remaining tricks greedily.

        Returns tricks won by declarer's side.
        """
        dec_tricks = 0

        # Complete the current trick
        if current_trick and current_trick.cards:
            # Cards already played in this trick
            trick_cards = dict(current_trick.cards)
            trick_cards[first_seat] = first_card
            leader = current_trick.leader
            order = [(leader + i) % 4 for i in range(4)]
            for seat in order:
                if seat in trick_cards:
                    continue
                # Greedy: play highest winning card or lowest loser
                h = hands.get(seat, [])
                led = current_trick.led_suit()
                following = [c for c in h if c.suit == led]
                if not following:
                    following = h
                if not following:
                    continue
                # Play highest
                pick = max(following, key=lambda c: c.rank)
                trick_cards[seat] = pick
                if seat in hands and pick in hands[seat]:
                    hands[seat].remove(pick)

            # Determine winner
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
            order = [(leader + i) % 4 for i in range(4)]
            for seat in order[1:]:
                h = hands.get(seat, [])
                if not h:
                    continue
                led = first_card.suit
                following = [c for c in h if c.suit == led]
                if not following:
                    following = h
                pick = max(following, key=lambda c: c.rank)
                trick_cards[seat] = pick
                hands[seat].remove(pick)

            winner = self._greedy_trick_winner(trick_cards, leader, trump_suit)
            if winner % 2 == declarer % 2:
                dec_tricks += 1
            next_leader = winner

        # Play remaining tricks greedily
        for _ in range(12):  # max remaining tricks
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
                    # Leader picks highest card in best suit
                    pick = max(h, key=lambda c: c.rank)
                    led_suit = pick.suit
                else:
                    following = [c for c in h if c.suit == led_suit]
                    if not following:
                        following = h
                    pick = max(following, key=lambda c: c.rank)
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
