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

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any
from enum import Enum, auto

from card import Card, Suit, Rank, DECK
from play import Trick
from bridge_params import BridgeParams


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

    Initialized with the player's own hand and dummy's hand (if visible).
    Updated after each completed trick.
    """

    def __init__(self, my_hand: List[Card], dummy_hand: Optional[List[Card]],
                 my_seat: int, dummy_seat: int):
        self.my_seat = my_seat
        self.dummy_seat = dummy_seat
        self.played: Set[Card] = set()
        self.shown_void: Dict[int, Set[Suit]] = {i: set() for i in range(4)}

        # All cards in the deck
        all_cards = set(DECK)
        # Cards we know the location of
        known = set(my_hand)
        if dummy_hand:
            known |= set(dummy_hand)
        self.unknown = all_cards - known

    def update_trick(self, trick: Trick):
        """Record a completed trick."""
        led = trick.led_suit()
        for seat, card in trick.cards.items():
            self.played.add(card)
            self.unknown.discard(card)
            if led and card.suit != led:
                self.shown_void[seat].add(led)

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


# ---------------------------------------------------------------------------
# DeclarerPlan
# ---------------------------------------------------------------------------

@dataclass
class DeclarerPlan:
    """High-level plan for declarer play, computed once per hand."""
    winners: int = 0
    target: int = 0
    shortfall: int = 0
    finesse_suits: List[Suit] = field(default_factory=list)
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

            # Finesse: we hold K (or Q) but not A (or K), and can lead toward it
            ranks_held = {c.rank for c in combined}
            if Rank.KING in ranks_held and Rank.ACE not in ranks_held:
                finesses.append(suit)
            elif Rank.QUEEN in ranks_held and Rank.KING not in ranks_held:
                finesses.append(suit)

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
                if dum_len <= 1 and dum_trumps > 0:
                    ruffs += min(dum_trumps, 2 - dum_len)

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
            playable = min(max(len(my), len(dum)), len(combined))
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

        # Build plan if we don't have one
        if self.plan is None:
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

        # Finesse: lead toward honor in the other hand
        if self.plan.finesse_suits:
            for fsuit in self.plan.finesse_suits:
                other_has_honor = any(
                    c.suit == fsuit and c.rank in (Rank.KING, Rank.QUEEN)
                    for c in other_hand)
                if other_has_honor:
                    # Lead low in that suit from current hand
                    low_cards = sorted([c for c in valid if c.suit == fsuit],
                                       key=lambda c: c.rank)
                    if low_cards and low_cards[0].rank < Rank.QUEEN:
                        return low_cards[0]

        # Cash winners (established cards)
        if self.tracker:
            for c in sorted(valid, key=lambda c: c.rank, reverse=True):
                if c.suit != trump_suit and self.tracker.is_established(c, hand, dummy):
                    return c

        # Draw trumps if we have trump control
        if trump_suit:
            trump_cards = sorted([c for c in valid if c.suit == trump_suit],
                                 key=lambda c: c.rank, reverse=True)
            if trump_cards and trump_cards[0].rank >= self.params.trump_draw_min:
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

        trick = obs.get('current_trick')
        declarer = obs.get('declarer')
        dummy_seat = (declarer + 2) % 4 if declarer is not None else -1
        is_declarer_side = (self.seat % 2 == (declarer or 0) % 2)
        on_lead = not trick or not trick.cards
        completed = obs.get('completed_tricks', [])

        if on_lead:
            if not is_declarer_side and len(completed) == 0:
                return self._opening_lead(obs)
            if is_declarer_side:
                return self._declarer_lead(obs)
            return self._defender_lead(obs)

        if is_declarer_side:
            return self._declarer_follow(obs)
        return self._defender_follow(obs)
