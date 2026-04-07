"""Hand evaluation utilities for contract bridge.

Pure functions for assessing hand strength, shape, and features.
All functions operate on ``list[Card]`` and depend only on ``card.py``.

Key concepts:
    - **HCP** (High Card Points): Milton Work 4-3-2-1 scale.
    - **Distribution points**: shortness values added to HCP for suit contracts.
    - **Support points**: revalued distribution when an 8+ card fit is confirmed.
    - **Losing Trick Count (LTC)**: alternative strength measure counting expected losers.
    - **Quick tricks**: immediate winners (AK=2, AQ=1.5, A=1, KQ=1, Kx=0.5).
"""

from dataclasses import dataclass
from typing import Optional, List
from card import Card, Suit, Rank


# ---------------------------------------------------------------------------
# HandShape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandShape:
    """Distribution summary of a bridge hand.

    Attributes:
        spades: Number of spades held.
        hearts: Number of hearts held.
        diamonds: Number of diamonds held.
        clubs: Number of clubs held.
    """
    spades: int
    hearts: int
    diamonds: int
    clubs: int

    @property
    def is_balanced(self) -> bool:
        """True for 4-3-3-3, 4-4-3-2, or 5-3-3-2 distributions."""
        s = self.shape_tuple
        return s in ((4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2))

    @property
    def longest_suit(self) -> Suit:
        """Suit with the most cards.  Ties broken S > H > D > C."""
        counts = {Suit.S: self.spades, Suit.H: self.hearts,
                  Suit.D: self.diamonds, Suit.C: self.clubs}
        return max(counts, key=lambda s: (counts[s], s))

    @property
    def second_suit(self) -> Optional[Suit]:
        """Second-longest suit with 4+ cards, or None."""
        longest = self.longest_suit
        counts = {Suit.S: self.spades, Suit.H: self.hearts,
                  Suit.D: self.diamonds, Suit.C: self.clubs}
        candidates = {s: c for s, c in counts.items() if s != longest and c >= 4}
        if not candidates:
            return None
        return max(candidates, key=lambda s: (candidates[s], s))

    @property
    def shape_tuple(self) -> tuple:
        """Suit lengths sorted descending, e.g. (5, 4, 3, 1)."""
        return tuple(sorted([self.spades, self.hearts, self.diamonds, self.clubs],
                            reverse=True))

    def length(self, suit: Suit) -> int:
        """Return the count for a specific suit.  Returns 0 for NT."""
        return {Suit.S: self.spades, Suit.H: self.hearts,
                Suit.D: self.diamonds, Suit.C: self.clubs}.get(suit, 0)


# ---------------------------------------------------------------------------
# Pure evaluation functions
# ---------------------------------------------------------------------------

def hcp(hand: List[Card]) -> int:
    """Milton Work high-card points: A=4, K=3, Q=2, J=1."""
    return sum(c.hcp() for c in hand)


def suit_length(hand: List[Card], suit: Suit) -> int:
    """Number of cards held in *suit*."""
    return sum(1 for c in hand if c.suit == suit)


def hand_shape(hand: List[Card]) -> HandShape:
    """Build a HandShape from a 13-card hand."""
    return HandShape(
        spades=suit_length(hand, Suit.S),
        hearts=suit_length(hand, Suit.H),
        diamonds=suit_length(hand, Suit.D),
        clubs=suit_length(hand, Suit.C),
    )


def distribution_points(hand: List[Card], params=None) -> int:
    """Shortness points for opener: void=3, singleton=2, doubleton=1."""
    void_pts = params.dist_void if params is not None else 3
    sing_pts = params.dist_singleton if params is not None else 2
    doub_pts = params.dist_doubleton if params is not None else 1
    pts = 0
    for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
        n = suit_length(hand, suit)
        if n == 0:
            pts += void_pts
        elif n == 1:
            pts += sing_pts
        elif n == 2:
            pts += doub_pts
    return pts


def support_points(hand: List[Card], fit_suit: Suit, params=None) -> int:
    """Revalued shortness when an 8+ card fit is confirmed.

    void=5, singleton=3, doubleton=1.  Only counts shortness
    outside the agreed fit suit.
    """
    void_pts = params.support_void if params is not None else 5
    sing_pts = params.support_singleton if params is not None else 3
    doub_pts = params.support_doubleton if params is not None else 1
    pts = 0
    for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
        if suit == fit_suit:
            continue
        n = suit_length(hand, suit)
        if n == 0:
            pts += void_pts
        elif n == 1:
            pts += sing_pts
        elif n == 2:
            pts += doub_pts
    return pts


def total_points(hand: List[Card], fit_suit: Optional[Suit] = None, params=None) -> int:
    """HCP plus distribution or support points.

    Uses support_points when *fit_suit* is provided, otherwise
    distribution_points.
    """
    h = hcp(hand)
    if fit_suit is not None:
        return h + support_points(hand, fit_suit, params=params)
    return h + distribution_points(hand, params=params)


def losing_trick_count(hand: List[Card]) -> int:
    """Losing Trick Count (LTC).

    For each suit, count up to 3 losers among the top 3 cards:
    - Ace = not a loser
    - King = not a loser if suit length >= 2
    - Queen = not a loser if suit length >= 3
    - Everything else in the top 3 slots = loser
    Void = 0 losers.
    """
    losers = 0
    for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
        cards = sorted([c for c in hand if c.suit == suit],
                       key=lambda c: c.rank, reverse=True)
        n = len(cards)
        if n == 0:
            continue
        check = min(n, 3)
        for i in range(check):
            rank = cards[i].rank
            if rank == Rank.ACE:
                continue  # not a loser
            if rank == Rank.KING and n >= 2:
                continue  # protected king
            if rank == Rank.QUEEN and n >= 3:
                continue  # protected queen
            losers += 1
    return losers


def quick_tricks(hand: List[Card]) -> float:
    """Immediate winners per suit: AK=2, AQ=1.5, A=1, KQ=1, Kx=0.5."""
    qt = 0.0
    for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
        cards = sorted([c for c in hand if c.suit == suit],
                       key=lambda c: c.rank, reverse=True)
        n = len(cards)
        if n == 0:
            continue
        has_a = n >= 1 and cards[0].rank == Rank.ACE
        has_k = n >= 2 and cards[1].rank == Rank.KING
        has_q = n >= 2 and cards[1].rank == Rank.QUEEN

        if has_a and has_k:
            qt += 2.0
        elif has_a and has_q:
            qt += 1.5
        elif has_a:
            qt += 1.0
        elif n >= 1 and cards[0].rank == Rank.KING:
            # K is top card
            if n >= 2 and cards[1].rank == Rank.QUEEN:
                qt += 1.0
            elif n >= 2:
                qt += 0.5
    return qt


def suit_quality(hand: List[Card], suit: Suit) -> int:
    """Count of honors (A, K, Q, J, T) held in *suit*."""
    honors = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN}
    return sum(1 for c in hand if c.suit == suit and c.rank in honors)


def stopper(hand: List[Card], suit: Suit) -> bool:
    """True if the hand has a stopper in *suit*: A, Kx+, Qxx+, or Jxxx+."""
    cards = [c for c in hand if c.suit == suit]
    n = len(cards)
    if n == 0:
        return False
    ranks = {c.rank for c in cards}
    if Rank.ACE in ranks:
        return True
    if Rank.KING in ranks and n >= 2:
        return True
    if Rank.QUEEN in ranks and n >= 3:
        return True
    if Rank.JACK in ranks and n >= 4:
        return True
    return False


def all_suits_stopped(hand: List[Card]) -> bool:
    """True if every suit (C, D, H, S) has a stopper."""
    return all(stopper(hand, s) for s in (Suit.C, Suit.D, Suit.H, Suit.S))


def rule_of_20(hand: List[Card]) -> bool:
    """True if HCP + two longest suits >= 20 (light opening guideline)."""
    h = hcp(hand)
    shape = hand_shape(hand)
    top2 = shape.shape_tuple[0] + shape.shape_tuple[1]
    return h + top2 >= 20


def biddable_suit(hand: List[Card], suit: Suit) -> bool:
    """True if the suit is long enough to bid: 5+ for majors, 4+ or 3+ for minors."""
    n = suit_length(hand, suit)
    if suit in (Suit.H, Suit.S):
        return n >= 5
    return n >= 3
