"""Core card primitives for a contract-bridge simulator.

Provides the fundamental building blocks used throughout the engine:

Classes:
    Suit: IntEnum encoding the four bridge suits plus no-trump (C=0, D=1,
        H=2, S=3, NT=4).  Numeric ordering matches standard bridge suit rank.
    Rank: IntEnum encoding card face values from TWO=2 through ACE=14.
        The integer values double as natural comparison keys so that
        ``Card`` objects sort correctly without extra logic.
    Card: Immutable, hashable dataclass combining a ``Rank`` and a ``Suit``.
        Suitable for use in sets and as dict keys.

Constants:
    DECK: The canonical 52-card deck — all (rank, suit) combinations for
        the four real suits, in a stable, deterministic order.

Functions:
    deal: Shuffle a copy of ``DECK`` and partition it into four 13-card
        hands, keyed 0–3.
"""

from enum import IntEnum
from dataclasses import dataclass
import random

class Suit(IntEnum):
    """The four bridge suits plus a no-trump sentinel, ordered by convention.

    Integer values follow standard bridge suit ranking (lowest to highest):
    Clubs (0), Diamonds (1), Hearts (2), Spades (3).  No-trump (4) is
    included as a sentinel for bidding logic; it is never assigned to a
    ``Card``.

    Members:
        C:  Clubs    (0)
        D:  Diamonds (1)
        H:  Hearts   (2)
        S:  Spades   (3)
        NT: No-trump (4) — bidding sentinel only, not a real suit.
    """
    C = 0
    D = 1
    H = 2
    S = 3
    NT = 4

class Rank(IntEnum):
    """Face value of a playing card, from TWO through ACE.

    Integer values equal the natural pip count for numbered cards and
    continue upward for honours (JACK=11, QUEEN=12, KING=13, ACE=14),
    making numeric comparison equivalent to card-rank comparison with no
    additional mapping required.

    Members:
        TWO–TEN: Spot cards (2–10).
        JACK:    11
        QUEEN:   12
        KING:    13
        ACE:     14 (highest)
    """
    TWO = 2; THREE = 3; FOUR = 4; FIVE = 5; SIX = 6
    SEVEN = 7; EIGHT = 8; NINE = 9; TEN = 10
    JACK = 11; QUEEN = 12; KING = 13; ACE = 14

SUIT_SYM = {Suit.C: '♣', Suit.D: '♦', Suit.H: '♥', Suit.S: '♠', Suit.NT: 'NT'}
RANK_SYM = {Rank.TWO:'2',Rank.THREE:'3',Rank.FOUR:'4',Rank.FIVE:'5',
            Rank.SIX:'6',Rank.SEVEN:'7',Rank.EIGHT:'8',Rank.NINE:'9',
            Rank.TEN:'T',Rank.JACK:'J',Rank.QUEEN:'Q',Rank.KING:'K',Rank.ACE:'A'}

@dataclass(frozen=True, order=True)
class Card:
    """An immutable, ordered playing card defined by its rank and suit.

    Implemented as a frozen dataclass so instances are hashable and can
    be stored in sets or used as dict keys.  The ``order=True`` flag
    delegates comparison to the (rank, suit) tuple, which means cards
    sort first by rank then by suit — matching the field declaration
    order.

    Attributes:
        rank: The card's face value as a ``Rank`` IntEnum member.
        suit: The card's suit as a ``Suit`` IntEnum member (never ``NT``).

    Example:
        >>> Card(Rank.ACE, Suit.S)
        A♠
        >>> Card(Rank.ACE, Suit.S).hcp()
        4
    """
    rank: Rank
    suit: Suit
    def __repr__(self): return f"{RANK_SYM[self.rank]}{SUIT_SYM[self.suit]}"
    def hcp(self):
        """Return the Milton Work high-card point (HCP) value of this card.

        Uses the standard 4-3-2-1 scale universally applied in bridge
        valuation:

        =========  ===
        Honour     HCP
        =========  ===
        Ace        4
        King       3
        Queen      2
        Jack       1
        All others 0
        =========  ===

        Returns:
            int: HCP value in the range [0, 4].
        """
        return {Rank.ACE:4,Rank.KING:3,Rank.QUEEN:2,Rank.JACK:1}.get(self.rank,0)

DECK = [Card(r, s) for s in list(Suit)[:4] for r in Rank]
"""list[Card]: The complete 52-card deck in canonical order.

Built once at import time as the Cartesian product of the four real suits
(Clubs, Diamonds, Hearts, Spades) with all 13 ranks.  Never mutated
directly; callers must copy before shuffling (see ``deal``).
"""

def deal():
    """Shuffle the deck and deal four sorted 13-card hands.

    Creates a fresh copy of ``DECK`` on every call, shuffles it in-place
    using ``random.shuffle`` (seeded by the OS entropy source by default),
    then slices the shuffled list into four equal partitions.  Each
    partition is sorted before being returned, giving hands in canonical
    (rank, suit) order for convenient display and comparison.

    Returns:
        dict[int, list[Card]]: Mapping of seat index to hand, where keys
        are 0 (South), 1 (West), 2 (North), 3 (East) by convention.
        Each value is a sorted list of exactly 13 ``Card`` objects.

    Example:
        >>> hands = deal()
        >>> len(hands[0])
        13
        >>> sum(len(h) for h in hands.values())
        52
    """
    d = DECK[:]
    random.shuffle(d)
    return {0: sorted(d[:13]), 1: sorted(d[13:26]), 2: sorted(d[26:39]), 3: sorted(d[39:])}
