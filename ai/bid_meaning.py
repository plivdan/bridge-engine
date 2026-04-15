"""Centralized bid-meaning classification.

Provides a single source of truth that the bidder, partner-estimator, and
fit-detection all read from. Without this, a convention added in one place
(e.g. bidding Jacoby Transfer) can drift from its interpretation in another
(e.g. reading partner's transfer as a natural diamond bid).

Scope today: responses to 1NT and 2NT openings. Expands per batch.
"""

from dataclasses import dataclass
from typing import Optional

from engine.card import Suit
from engine.auction import Bid


@dataclass
class BidMeaning:
    convention: str = 'natural'
    shows_suit: Optional[Suit] = None
    shows_length_min: int = 0
    hcp_min: int = 0
    hcp_max: int = 40
    is_transfer: bool = False
    promises_4_major: bool = False
    is_invitational: bool = False
    is_signoff: bool = False
    asks_aces: bool = False
    is_balanced: Optional[bool] = None


def interpret_response_to_1nt(call: Bid, params) -> BidMeaning:
    """Classify partner's response to a 1NT opening (15-17 balanced)."""
    if call is None or call.special:
        return BidMeaning(convention='pass')
    lv, st = call.level, call.strain

    if lv == 2 and st == Suit.C and params.use_stayman:
        return BidMeaning(
            convention='stayman',
            hcp_min=params.stayman_min_hcp,
            promises_4_major=True,
        )
    if lv == 2 and st == Suit.D and params.use_jacoby_transfers:
        return BidMeaning(
            convention='jacoby_transfer',
            shows_suit=Suit.H,
            shows_length_min=5,
            is_transfer=True,
        )
    if lv == 2 and st == Suit.H and params.use_jacoby_transfers:
        return BidMeaning(
            convention='jacoby_transfer',
            shows_suit=Suit.S,
            shows_length_min=5,
            is_transfer=True,
        )
    if lv == 4 and st == Suit.C and params.use_gerber:
        return BidMeaning(
            convention='gerber',
            asks_aces=True,
            hcp_min=params.gerber_min_hcp,
        )
    if lv == 2 and st == Suit.NT:
        return BidMeaning(
            convention='natural', hcp_min=8, hcp_max=9,
            is_invitational=True, is_balanced=True,
        )
    if lv == 3 and st == Suit.NT:
        return BidMeaning(
            convention='natural', hcp_min=10, hcp_max=15, is_balanced=True,
        )
    if lv == 4 and st == Suit.NT:
        return BidMeaning(
            convention='quantitative', hcp_min=16, hcp_max=17, is_balanced=True,
        )
    if lv == 3 and st in (Suit.C, Suit.D, Suit.H, Suit.S):
        return BidMeaning(
            convention='natural', shows_suit=st,
            shows_length_min=6, hcp_min=10,
        )
    if lv == 4 and st in (Suit.H, Suit.S):
        return BidMeaning(
            convention='natural', shows_suit=st,
            shows_length_min=6, hcp_min=10, is_signoff=True,
        )
    if lv == 2 and st == Suit.S:
        # 2S natural signoff (weak hand with long spades, no transfer available
        # if hearts-to-spades transfer is already taken by 2H)
        return BidMeaning(
            convention='natural', shows_suit=Suit.S,
            shows_length_min=5, hcp_min=0, hcp_max=7, is_signoff=True,
        )
    return BidMeaning(
        convention='natural',
        shows_suit=st if st != Suit.NT else None,
    )


def gerber_ace_response(num_aces: int) -> Suit:
    """Map ace count to the 4-of-a-suit response strain."""
    return {0: Suit.D, 1: Suit.H, 2: Suit.S, 3: Suit.NT, 4: Suit.D}[num_aces]


def gerber_aces_from_response(call: Bid) -> Optional[int]:
    """Decode a 4-of-a-suit Gerber response into an ace count (0-4).

    4D = 0 or 4 (ambiguous; caller must decide from context).
    4H = 1, 4S = 2, 4NT = 3.
    """
    if call is None or call.special or call.level != 4:
        return None
    if call.strain == Suit.D:
        return 0  # caller infers 4 if combined strength demands it
    if call.strain == Suit.H:
        return 1
    if call.strain == Suit.S:
        return 2
    if call.strain == Suit.NT:
        return 3
    return None
