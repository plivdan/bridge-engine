"""Per-seat HCP and shape inference from an auction.

Shared between declarer play, defender counting, and the Monte Carlo
dealer. Before this module, the MC sampler generated uniformly random
deals that routinely violated what the auction told us (e.g., sampling
a 4-HCP hand for an opponent who opened 1NT); declarer had no framework
for placing a missing queen based on who opened.

The inference is deliberately coarse — it captures the constraints
any competent player would read from the auction, not subtle
inferences like "partner would have bid X over Y if they held Z."
Richer inference from played cards (count signals, discards) lives
in CardTracker.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.card import Suit
from engine.auction import Bid, PASS


@dataclass
class SeatConstraints:
    """HCP and per-suit length bounds for one seat."""
    hcp_min: int = 0
    hcp_max: int = 40
    suit_min: Dict[Suit, int] = field(
        default_factory=lambda: {s: 0 for s in
                                 (Suit.S, Suit.H, Suit.D, Suit.C)})
    suit_max: Dict[Suit, int] = field(
        default_factory=lambda: {s: 13 for s in
                                 (Suit.S, Suit.H, Suit.D, Suit.C)})
    is_balanced: Optional[bool] = None

    def tighten_hcp(self, lo: int, hi: int):
        self.hcp_min = max(self.hcp_min, lo)
        self.hcp_max = min(self.hcp_max, hi)

    def set_suit_min(self, suit: Suit, n: int):
        if suit not in self.suit_min:
            return  # NT or other non-suit strain — no length to set
        self.suit_min[suit] = max(self.suit_min[suit], n)

    def set_suit_max(self, suit: Suit, n: int):
        if suit not in self.suit_max:
            return
        self.suit_max[suit] = min(self.suit_max[suit], n)

    def hand_is_consistent(self, hand) -> bool:
        """Check if a candidate hand satisfies these constraints."""
        from .hand_eval import hcp as _hcp
        h = _hcp(hand)
        if not (self.hcp_min <= h <= self.hcp_max):
            return False
        for s in (Suit.S, Suit.H, Suit.D, Suit.C):
            length = sum(1 for c in hand if c.suit == s)
            if not (self.suit_min[s] <= length <= self.suit_max[s]):
                return False
        if self.is_balanced is True:
            lengths = sorted(
                [sum(1 for c in hand if c.suit == s)
                 for s in (Suit.S, Suit.H, Suit.D, Suit.C)])
            # Balanced = 4-3-3-3, 4-4-3-2, or 5-3-3-2
            if lengths[0] < 2 or lengths[-1] > 5:
                return False
            if lengths == [2, 2, 4, 5] or lengths == [2, 2, 3, 6]:
                return False
        return True


def infer_from_auction(calls: List[Bid], dealer: int,
                       params=None) -> Dict[int, SeatConstraints]:
    """Infer per-seat constraints from the auction so far.

    Only the first non-special call per seat is treated as a
    show-of-hand (opening or overcall); subsequent bids are not
    interpreted here to keep the module self-contained. The bidding
    agent carries its own partner-specific follow-up inference.
    """
    from .bridge_params import BridgeParams
    p = params or BridgeParams()

    out = {s: SeatConstraints() for s in range(4)}
    seat_has_natural = {s: False for s in range(4)}
    seat_passed_before_natural = {s: False for s in range(4)}

    for i, call in enumerate(calls):
        seat = (dealer + i) % 4
        c = out[seat]

        if call.special:
            # Track whether this seat passed before ever bidding naturally
            if not seat_has_natural[seat] and call == PASS:
                seat_passed_before_natural[seat] = True
            continue

        is_first_natural = not seat_has_natural[seat]
        seat_has_natural[seat] = True

        if not is_first_natural:
            continue  # subsequent bids: not modeled here

        # 1NT opening
        if call.level == 1 and call.strain == Suit.NT:
            c.tighten_hcp(p.open_1nt_min, p.open_1nt_max)
            c.is_balanced = True
            for s in (Suit.S, Suit.H, Suit.D, Suit.C):
                c.set_suit_min(s, 2)
                c.set_suit_max(s, 5)
            continue
        # 2NT opening
        if call.level == 2 and call.strain == Suit.NT:
            c.tighten_hcp(p.open_2nt_min, p.open_2nt_max)
            c.is_balanced = True
            for s in (Suit.S, Suit.H, Suit.D, Suit.C):
                c.set_suit_min(s, 2)
                c.set_suit_max(s, 5)
            continue
        # 2C strong
        if call.level == 2 and call.strain == Suit.C:
            c.tighten_hcp(p.open_strong_min, 40)
            continue
        # 1-of-suit opening
        if call.level == 1:
            c.tighten_hcp(p.open_min_hcp, 21)
            if call.strain in (Suit.H, Suit.S):
                c.set_suit_min(call.strain, 5)
            else:
                c.set_suit_min(call.strain, 3)
            continue
        # Weak 2 (D/H/S)
        if call.level == 2 and call.strain in (Suit.D, Suit.H, Suit.S):
            c.tighten_hcp(p.weak_two_min_hcp, p.weak_two_max_hcp)
            c.set_suit_min(call.strain, 6)
            c.set_suit_max(call.strain, 6)
            continue
        # 3-level preempt
        if call.level == 3:
            c.tighten_hcp(p.preempt_3_min_hcp, p.preempt_3_max_hcp)
            c.set_suit_min(call.strain, 7)
            continue
        # 4-level major preempt
        if call.level == 4 and call.strain in (Suit.H, Suit.S):
            c.tighten_hcp(p.preempt_4_min_hcp, p.preempt_4_max_hcp)
            c.set_suit_min(call.strain, 8)
            continue

    # Passed-hand upper bound: a seat that passed before ever bidding
    # naturally (in a non-special auction action) denies opening values.
    for s in range(4):
        if seat_passed_before_natural[s] and not seat_has_natural[s]:
            out[s].tighten_hcp(0, p.open_min_hcp - 1)

    return out
