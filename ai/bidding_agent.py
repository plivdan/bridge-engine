"""State-machine bidding agent implementing Standard American.

The bidder recomputes its phase from the auction history on every call
to ``bid()``, so it carries no mutable state between calls.

Bidding system summary (Standard American Yellow Card):
    - 1NT opening: 15-17 HCP, balanced, all suits stopped
    - 2NT opening: 20-21 HCP, balanced
    - 2C opening: 22+ HCP (artificial, strong, forcing)
    - 1-of-suit: 12-21 HCP, 5+ in majors, 3+ in minors
    - Responses calibrated to reach game with 26+ combined points
    - Blackwood 4NT for slam investigation
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from engine.card import Card, Suit, Rank
from engine.auction import Bid, PASS, DOUBLE, REDOUBLE, make_bid
from .hand_eval import (
    hcp, hand_shape, total_points, distribution_points, support_points,
    suit_length, suit_quality, stopper, all_suits_stopped, rule_of_20,
    biddable_suit, quick_tricks, HandShape,
)
from .bridge_params import BridgeParams
from .trace import DecisionTrace


class BidPhase(Enum):
    """Which stage of the bidding conversation we are in."""
    OPENING = auto()
    OVERCALL = auto()
    RESPONDING = auto()
    OPENER_REBID = auto()
    RESPONDER_REBID = auto()
    COMPETITIVE = auto()
    SLAM_INVESTIGATION = auto()
    SIGNOFF = auto()


@dataclass
class PartnerEstimate:
    """Running estimate of partner's hand based on their bids."""
    min_hcp: int = 0
    max_hcp: int = 40
    known_suits: Dict[Suit, int] = field(default_factory=dict)
    is_balanced: Optional[bool] = None


class StateMachineBidder:
    """Standard American bidding agent.

    Args:
        seat: Seat index (0=N, 1=E, 2=S, 3=W).
    """

    def __init__(self, seat: int, params=None):
        self.seat = seat
        self.params = params or BridgeParams()
        self.last_trace: Optional[DecisionTrace] = None

    # ------------------------------------------------------------------
    # helpers to parse the auction history
    # ------------------------------------------------------------------

    def _my_bids(self, calls: list, dealer: int) -> List[Bid]:
        return [calls[i] for i in range(len(calls))
                if (dealer + i) % 4 == self.seat and not calls[i].special]

    def _partner_bids(self, calls: list, dealer: int) -> List[Bid]:
        partner = (self.seat + 2) % 4
        return [calls[i] for i in range(len(calls))
                if (dealer + i) % 4 == partner and not calls[i].special]

    def _opp_bids(self, calls: list, dealer: int) -> List[Bid]:
        return [calls[i] for i in range(len(calls))
                if (dealer + i) % 4 % 2 != self.seat % 2 and not calls[i].special]

    def _partner_calls(self, calls: list, dealer: int) -> List[Bid]:
        partner = (self.seat + 2) % 4
        return [calls[i] for i in range(len(calls))
                if (dealer + i) % 4 == partner]

    def _last_opp_bid(self, calls: list, dealer: int) -> Optional[Bid]:
        for i in range(len(calls) - 1, -1, -1):
            if (dealer + i) % 4 % 2 != self.seat % 2 and not calls[i].special:
                return calls[i]
        return None

    # ------------------------------------------------------------------
    # phase detection
    # ------------------------------------------------------------------

    def _determine_phase(self, obs: dict) -> BidPhase:
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        my = self._my_bids(calls, dealer)
        partner = self._partner_bids(calls, dealer)
        opp = self._opp_bids(calls, dealer)

        if not my and not partner:
            return BidPhase.OVERCALL if opp else BidPhase.OPENING

        if partner and not my:
            return BidPhase.RESPONDING

        if my and partner:
            # check for slam potential — use conservative estimate
            fit = self._detected_fit(my, partner, obs['hand'])
            tp = total_points(obs['hand'], fit)
            est = self._estimate_partner(partner, calls, dealer)
            # For slam, use min + 1/3 of range (conservative)
            partner_est = est.min_hcp + int((est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
            combined = tp + partner_est
            if combined >= self.params.slam_small_min and fit is not None:
                return BidPhase.SLAM_INVESTIGATION
            if len(my) == 1 and len(partner) >= 1:
                return BidPhase.OPENER_REBID
            return BidPhase.RESPONDER_REBID

        if my and not partner:
            if opp:
                return BidPhase.COMPETITIVE
            return BidPhase.SIGNOFF

        return BidPhase.SIGNOFF

    # ------------------------------------------------------------------
    # fit and target
    # ------------------------------------------------------------------

    def _detected_fit(self, my_bids: list, partner_bids: list,
                      hand: List[Card]) -> Optional[Suit]:
        """Return best agreed trump suit if an 8+ card fit is likely.

        Checks all suits and returns the one with the longest combined
        holding, preferring majors over minors.
        """
        best_suit = None
        best_len = 0
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            my_len = suit_length(hand, suit)
            partner_bid_suit = any(b.strain == suit for b in partner_bids)
            i_bid_suit = any(b.strain == suit for b in my_bids)
            # Estimate combined length
            partner_min = 0
            if partner_bid_suit:
                partner_min = 4  # bid promises at least 4
            fit_found = False
            # partner bid the suit (promises 4+) and I have 4+
            if partner_bid_suit and my_len >= self.params.fit_min_support_general:
                fit_found = True
            # I bid the suit (promise 5+ major, 3+ minor) and partner raised
            if i_bid_suit and partner_bid_suit:
                fit_found = True
            # partner bid and I have 3+ support in a major
            if partner_bid_suit and my_len >= self.params.fit_min_support_major and suit in (Suit.H, Suit.S):
                fit_found = True

            if fit_found:
                combined = my_len + partner_min
                # Prefer longer fits; break ties by suit rank (majors first)
                if combined > best_len or (combined == best_len and best_suit is None):
                    best_len = combined
                    best_suit = suit
        return best_suit

    def _target_level(self, combined: int, fit: Optional[Suit],
                      hand: Optional[List[Card]] = None) -> tuple:
        """Map combined points to (level, strain) target."""
        if combined >= self.params.slam_grand_min:
            strain = fit if fit is not None else Suit.NT
            return (7, strain)
        if combined >= self.params.slam_small_min:
            strain = fit if fit is not None else Suit.NT
            return (6, strain)
        game_min = getattr(self, '_effective_game_min', self.params.game_combined_min)
        if combined >= game_min:
            if fit in (Suit.H, Suit.S):
                return (4, fit)
            if fit in (Suit.C, Suit.D):
                # Prefer 3NT with stoppers, otherwise 5 of the minor
                if hand and all_suits_stopped(hand):
                    return (3, Suit.NT)
                return (5, fit)
            return (3, Suit.NT)
        if combined >= self.params.inv_combined_min:
            return (2, Suit.NT)
        if fit is not None:
            return (1, fit)
        return (1, Suit.NT)

    # ------------------------------------------------------------------
    # partner estimation
    # ------------------------------------------------------------------

    def _estimate_partner(self, partner_bids: list, calls: list,
                          dealer: int) -> PartnerEstimate:
        est = PartnerEstimate()
        if not partner_bids:
            return est

        first = partner_bids[0]
        my_bids_here = self._my_bids(calls, dealer)

        # Did I bid before partner? If so, partner is RESPONDING (6-18),
        # not OPENING (12-21).  Check by comparing indices in the call list.
        partner_seat = (self.seat + 2) % 4
        my_first_idx = None
        partner_first_idx = None
        for i, c in enumerate(calls):
            seat_i = (dealer + i) % 4
            if seat_i == self.seat and not c.special and my_first_idx is None:
                my_first_idx = i
            if seat_i == partner_seat and not c.special and partner_first_idx is None:
                partner_first_idx = i

        partner_is_responder = (my_first_idx is not None and partner_first_idx is not None
                                and my_first_idx < partner_first_idx)

        if partner_is_responder:
            # Partner is RESPONDING to our opening
            if first.level == 1 and first.strain == Suit.NT:
                est.min_hcp, est.max_hcp = 6, 10  # 1NT response
                est.is_balanced = True
            elif first.level == 2 and first.strain == Suit.NT:
                est.min_hcp, est.max_hcp = 10, 12  # 2NT response (invitational)
                est.is_balanced = True
            elif first.level == 3 and first.strain == Suit.NT:
                est.min_hcp, est.max_hcp = 13, 15  # 3NT response
                est.is_balanced = True
            elif first.level == 1:
                est.min_hcp, est.max_hcp = 6, 18  # new suit at 1-level
                if first.strain in (Suit.H, Suit.S):
                    est.known_suits[first.strain] = 4
                else:
                    est.known_suits[first.strain] = 4
            elif first.level == 2:
                # Simple raise = 6-10, new suit at 2 = 10+
                if my_bids_here and first.strain == my_bids_here[0].strain:
                    est.min_hcp, est.max_hcp = 6, 10  # simple raise
                    est.known_suits[first.strain] = 3
                else:
                    est.min_hcp, est.max_hcp = 10, 18  # new suit at 2
                    est.known_suits[first.strain] = 4
            elif first.level == 3:
                # Limit raise = 10-12
                if my_bids_here and first.strain == my_bids_here[0].strain:
                    est.min_hcp, est.max_hcp = 10, 12
                    est.known_suits[first.strain] = 4
                else:
                    est.min_hcp, est.max_hcp = 10, 18
            elif first.level >= 4:
                # Game raise = 13+
                est.min_hcp, est.max_hcp = 13, 18
                if first.strain in (Suit.H, Suit.S):
                    est.known_suits[first.strain] = 4
        else:
            # Partner is OPENING
            if first.level == 1 and first.strain == Suit.NT:
                est.min_hcp, est.max_hcp = 15, 17
                est.is_balanced = True
            elif first.level == 2 and first.strain == Suit.NT:
                est.min_hcp, est.max_hcp = 20, 21
                est.is_balanced = True
            elif first.level == 2 and first.strain == Suit.C:
                est.min_hcp, est.max_hcp = 22, 40
            elif first.level == 1:
                est.min_hcp, est.max_hcp = 12, 21
                if first.strain in (Suit.H, Suit.S):
                    est.known_suits[first.strain] = 5
                else:
                    est.known_suits[first.strain] = 3
            elif first.level >= 2:
                est.min_hcp, est.max_hcp = 6, 18
                if first.strain in (Suit.H, Suit.S):
                    est.known_suits[first.strain] = 5

        # If partner raised our suit, they show support
        if my_bids_here:
            my_strain = my_bids_here[0].strain
            for pb in partner_bids:
                if pb.strain == my_strain:
                    est.known_suits[my_strain] = max(
                        est.known_suits.get(my_strain, 0), 3)
        return est

    # ------------------------------------------------------------------
    # select a valid bid closest to target
    # ------------------------------------------------------------------

    def _best_valid(self, target_level: int, target_strain: Suit,
                    valid: list) -> Optional[Bid]:
        """Return the bid at (target_level, target_strain) if legal."""
        target = make_bid(target_level, target_strain)
        if target in valid:
            return target
        return None

    def _bid_up_to(self, level: int, strain: Suit, valid: list) -> Bid:
        """Bid at (level, strain) if valid, else the cheapest legal bid
        in that strain, else PASS."""
        target = make_bid(level, strain)
        if target in valid:
            return target
        # try cheaper levels in the same strain
        for lv in range(1, 8):
            b = make_bid(lv, strain)
            if b in valid:
                return b
        return PASS

    def _cheapest_in_suit(self, suit: Suit, valid: list) -> Optional[Bid]:
        """Cheapest available bid in the given suit."""
        for lv in range(1, 8):
            b = make_bid(lv, suit)
            if b in valid:
                return b
        return None

    # ------------------------------------------------------------------
    # OPENING
    # ------------------------------------------------------------------

    def _bid_opening(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        h = hcp(hand)
        shape = hand_shape(hand)

        # Sub-opening
        if h < self._effective_open_min and not rule_of_20(hand):
            return PASS

        # Strong 2C
        if h >= self.params.open_strong_min:
            b = self._best_valid(2, Suit.C, valid)
            if b:
                return b
            # fallback: bid highest available
            return self._open_suit(hand, shape, valid)

        # 2NT: 20-21 balanced
        if self.params.open_2nt_min <= h <= self.params.open_2nt_max and shape.is_balanced:
            b = self._best_valid(2, Suit.NT, valid)
            if b:
                return b

        # 1NT: 15-17 balanced.  18-19 balanced opens 1-suit and plans
        # a jump rebid to 2NT, but we stretch to include them here for
        # simplicity — the rebid logic now handles extras properly.
        if self.params.open_1nt_min <= h <= self.params.open_1nt_max and shape.is_balanced:
            b = self._best_valid(1, Suit.NT, valid)
            if b:
                return b

        # 1-of-suit
        return self._open_suit(hand, shape, valid)

    def _open_suit(self, hand: List[Card], shape: HandShape,
                   valid: list) -> Bid:
        """Choose 1-of-suit opening."""
        # 5+ card major → bid it (prefer spades with 5-5)
        for suit in (Suit.S, Suit.H):
            if shape.length(suit) >= 5:
                b = self._cheapest_in_suit(suit, valid)
                if b:
                    return b

        # Longer minor
        d_len = shape.diamonds
        c_len = shape.clubs
        if d_len >= c_len and d_len >= 3:
            b = self._cheapest_in_suit(Suit.D, valid)
            if b:
                return b
        if c_len >= 3:
            b = self._cheapest_in_suit(Suit.C, valid)
            if b:
                return b

        # fallback: 1C (always biddable as a "better minor")
        b = self._cheapest_in_suit(Suit.C, valid)
        return b if b else PASS

    # ------------------------------------------------------------------
    # RESPONDING
    # ------------------------------------------------------------------

    def _bid_responding(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        h = hcp(hand)
        shape = hand_shape(hand)
        partner_bids = self._partner_bids(calls, dealer)

        if not partner_bids:
            return PASS

        opening = partner_bids[0]

        # Response to 1NT (15-17)
        if opening.level == 1 and opening.strain == Suit.NT:
            return self._respond_to_1nt(hand, h, valid)

        # Response to 2NT (20-21)
        if opening.level == 2 and opening.strain == Suit.NT:
            return self._respond_to_2nt(hand, h, valid)

        # Response to 2C (22+ strong)
        if opening.level == 2 and opening.strain == Suit.C:
            return self._respond_to_2c(hand, h, shape, valid)

        # Response to 1-of-suit
        if opening.level == 1:
            return self._respond_to_1_suit(hand, h, shape, opening, valid)

        return PASS

    def _respond_to_1nt(self, hand: List[Card], h: int,
                        valid: list) -> Bid:
        # 0-7: pass
        if h <= self.params.respond_1nt_pass_max:
            return PASS
        # 8-9: 2NT (invitational)
        if h <= self.params.respond_1nt_inv_max:
            b = self._best_valid(2, Suit.NT, valid)
            return b if b else PASS
        # 10-15: 3NT
        if h <= self.params.respond_1nt_game_max:
            b = self._best_valid(3, Suit.NT, valid)
            return b if b else PASS
        # 16+: 4NT (quantitative slam invite)
        b = self._best_valid(4, Suit.NT, valid)
        return b if b else self._best_valid(3, Suit.NT, valid) or PASS

    def _respond_to_2nt(self, hand: List[Card], h: int,
                        valid: list) -> Bid:
        # 0-3: pass
        if h <= self.params.respond_2nt_pass_max:
            return PASS
        # 4-10: 3NT
        if h <= self.params.respond_2nt_game_max:
            b = self._best_valid(3, Suit.NT, valid)
            return b if b else PASS
        # 11+: 4NT slam invite
        b = self._best_valid(4, Suit.NT, valid)
        return b if b else self._best_valid(3, Suit.NT, valid) or PASS

    def _respond_to_2c(self, hand: List[Card], h: int, shape: HandShape,
                       valid: list) -> Bid:
        # 2D waiting response (artificial, 0-7 HCP)
        if h <= self.params.respond_2c_weak_max:
            b = self._best_valid(2, Suit.D, valid)
            return b if b else PASS
        # 8+ with a 5+ suit: bid it
        for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
            if shape.length(suit) >= 5 and suit_quality(hand, suit) >= 2:
                b = self._cheapest_in_suit(suit, valid)
                if b:
                    return b
        # 2NT: 8+ balanced
        b = self._best_valid(2, Suit.NT, valid)
        return b if b else PASS

    def _respond_to_1_suit(self, hand: List[Card], h: int,
                           shape: HandShape, opening: Bid,
                           valid: list) -> Bid:
        strain = opening.strain

        # Too weak to respond
        if h < self.params.respond_min_hcp:
            return PASS

        # Collect opponent's bid suits to avoid raising them
        obs_calls = getattr(self, '_current_obs_calls', [])
        opp_suits = set()
        for i, c in enumerate(obs_calls):
            if not c.special and (getattr(self, '_current_dealer', 0) + i) % 4 % 2 != self.seat % 2:
                opp_suits.add(c.strain)

        # Major fit: raise partner's suit (not opponent's!)
        if strain in (Suit.H, Suit.S) and strain not in opp_suits:
            support = suit_length(hand, strain)
            if support >= 4 and h >= self.params.respond_raise_game_min:
                b = self._best_valid(4, strain, valid)
                if b:
                    return b
            if support >= 4 and h >= self.params.respond_raise_limit_min:
                b = self._best_valid(3, strain, valid)
                if b:
                    return b
            if support >= 3 and h >= self.params.respond_min_hcp:
                b = self._best_valid(2, strain, valid)
                if b:
                    return b

        # New suit at 1 level (6+ HCP), avoid opponent's suits
        for suit in (Suit.S, Suit.H):
            if suit > strain and suit not in opp_suits and shape.length(suit) >= 4:
                b = self._cheapest_in_suit(suit, valid)
                if b and b.level == 1:
                    return b

        # 1NT response (6-10, no fit, no new suit at 1)
        if self.params.respond_min_hcp <= h <= self.params.respond_raise_limit_min:
            b = self._best_valid(1, Suit.NT, valid)
            if b:
                return b

        # New suit at 2 level (10+ HCP), avoid opponent's suits
        if h >= self.params.respond_new_2_min:
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit != strain and suit not in opp_suits and shape.length(suit) >= 4:
                    b = self._cheapest_in_suit(suit, valid)
                    if b:
                        return b

        # 2NT (10-12, no fit)
        if self.params.respond_2nt_inv_min <= h <= self.params.respond_2nt_inv_max:
            b = self._best_valid(2, Suit.NT, valid)
            if b:
                return b

        # 3NT (13+, stoppers)
        if h >= self.params.respond_3nt_min and all_suits_stopped(hand):
            b = self._best_valid(3, Suit.NT, valid)
            if b:
                return b

        # 2NT as fallback with 13+ (even without all stoppers)
        if h >= self.params.respond_3nt_min:
            b = self._best_valid(2, Suit.NT, valid)
            if b:
                return b

        # Fallback: raise partner or pass
        return PASS

    # ------------------------------------------------------------------
    # OPENER REBID
    # ------------------------------------------------------------------

    def _bid_opener_rebid(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        h = hcp(hand)
        shape = hand_shape(hand)
        partner_bids = self._partner_bids(calls, dealer)
        my_bids = self._my_bids(calls, dealer)

        if not my_bids or not partner_bids:
            return PASS

        my_opening = my_bids[0]
        partner_resp = partner_bids[0]
        fit = self._detected_fit(my_bids, partner_bids, hand)
        est = self._estimate_partner(partner_bids, calls, dealer)
        combined_min = h + est.min_hcp
        combined_max = h + est.max_hcp

        # If game is already reached, don't bid further unless slam values
        current_contract = obs.get('contract')
        if current_contract and not current_contract.special:
            clv = current_contract.level
            game_reached = (
                (clv >= 3 and current_contract.strain == Suit.NT) or
                (clv >= 4 and current_contract.strain in (Suit.H, Suit.S)) or
                (clv >= 5 and current_contract.strain in (Suit.C, Suit.D))
            )
            tp_est = total_points(hand, fit) + est.min_hcp + int(
                (est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
            if game_reached and tp_est < self.params.slam_small_min:
                return PASS

        # If we have a fit, raise to the appropriate level
        if fit:
            tp = total_points(hand, fit)
            # Use conservative partner estimate for rebids
            partner_est = est.min_hcp + int((est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
            combined = tp + partner_est
            # Cap at game level — slam decisions go through slam phase
            target_lv, target_st = self._target_level(min(combined, 32), fit, hand)
            b = self._best_valid(target_lv, target_st, valid)
            if not b:
                # Target level too low for current auction — raise to cheapest available
                b = self._cheapest_in_suit(fit, valid)
            if b:
                return b

        # --- Minimum opener (12-14): rebid cheaply ---
        if h <= self.params.rebid_min_max:
            # rebid 6+ card suit
            if shape.length(my_opening.strain) >= 6:
                b = self._cheapest_in_suit(my_opening.strain, valid)
                if b:
                    return b
            # 1NT rebid (balanced-ish)
            b = self._best_valid(1, Suit.NT, valid)
            if b:
                return b
            # new suit at cheapest level
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit != my_opening.strain and shape.length(suit) >= 4:
                    b = self._cheapest_in_suit(suit, valid)
                    if b and b.level <= 2:
                        return b
            # rebid own suit at cheapest level
            b = self._cheapest_in_suit(my_opening.strain, valid)
            if b:
                return b
            return PASS

        # --- Medium opener (15-17): show extras ---
        if h <= self.params.rebid_med_max:
            # jump rebid own suit with 6+
            if shape.length(my_opening.strain) >= 6:
                for lv in range(2, 4):
                    b = make_bid(lv, my_opening.strain)
                    if b in valid:
                        return b
            # 2NT (shows 15-17, balanced-ish)
            b = self._best_valid(2, Suit.NT, valid)
            if b:
                return b
            # new suit (forcing, shows extras)
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit != my_opening.strain and shape.length(suit) >= 4:
                    b = self._cheapest_in_suit(suit, valid)
                    if b:
                        return b
            # rebid own suit
            b = self._cheapest_in_suit(my_opening.strain, valid)
            if b:
                return b
            return PASS

        # --- Strong (18-19): jump or bid game ---
        if h <= self.params.rebid_strong_max:
            if fit:
                target_lv, target_st = self._target_level(
                    total_points(hand, fit) + est.min_hcp, fit, hand)
                b = self._best_valid(target_lv, target_st, valid)
                if b:
                    return b
            # 3NT with stoppers
            if all_suits_stopped(hand):
                b = self._best_valid(3, Suit.NT, valid)
                if b:
                    return b
            # 2NT
            b = self._best_valid(2, Suit.NT, valid)
            if b:
                return b
            # jump in own suit
            if shape.length(my_opening.strain) >= 6:
                b = self._best_valid(3, my_opening.strain, valid)
                if b:
                    return b
            # new suit
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if suit != my_opening.strain and shape.length(suit) >= 4:
                    b = self._cheapest_in_suit(suit, valid)
                    if b:
                        return b
            b = self._cheapest_in_suit(my_opening.strain, valid)
            if b:
                return b
            return PASS

        # --- Very strong (20+): bid game ---
        if fit:
            target_lv, target_st = self._target_level(
                total_points(hand, fit) + est.min_hcp, fit, hand)
            b = self._best_valid(target_lv, target_st, valid)
            if b:
                return b
        if all_suits_stopped(hand):
            b = self._best_valid(3, Suit.NT, valid)
            if b:
                return b
        b = self._best_valid(2, Suit.NT, valid)
        if b:
            return b
        b = self._cheapest_in_suit(my_opening.strain, valid)
        return b if b else PASS

    # ------------------------------------------------------------------
    # RESPONDER REBID
    # ------------------------------------------------------------------

    def _bid_responder_rebid(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        h = hcp(hand)
        shape = hand_shape(hand)
        my_bids = self._my_bids(calls, dealer)
        partner_bids = self._partner_bids(calls, dealer)
        fit = self._detected_fit(my_bids, partner_bids, hand)
        est = self._estimate_partner(partner_bids, calls, dealer)
        partner_est = est.min_hcp + int((est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
        combined = total_points(hand, fit) + partner_est

        # Weak hands (< 10 HCP): pass unless we have a clear fit at a safe level
        if h < self.params.responder_rebid_weak_max:
            if fit and combined >= self.params.responder_rebid_fit_min_combined:
                # Raise partner's suit at cheapest level
                b = self._cheapest_in_suit(fit, valid)
                if b and b.level <= 3:
                    return b
            return PASS

        # If game is already reached, don't bid further unless slam values
        current_contract = obs.get('contract')
        if current_contract and not current_contract.special:
            contract_lv = current_contract.level
            # Game is reached at 3NT, 4M, 5m
            game_reached = (
                (contract_lv >= 3 and current_contract.strain == Suit.NT) or
                (contract_lv >= 4 and current_contract.strain in (Suit.H, Suit.S)) or
                (contract_lv >= 5 and current_contract.strain in (Suit.C, Suit.D))
            )
            if game_reached and combined < self.params.slam_small_min:
                return PASS

        # Cap at game — slam goes through slam phase
        target_lv, target_st = self._target_level(min(combined, 32), fit, hand)

        # Try to bid the target
        b = self._best_valid(target_lv, target_st, valid)
        if b:
            return b

        # If target not available, try game in a major
        if fit in (Suit.H, Suit.S) and combined >= self.params.game_combined_min:
            b = self._best_valid(4, fit, valid)
            if b:
                return b

        # 3NT with values
        if combined >= self.params.game_combined_min and all_suits_stopped(hand):
            b = self._best_valid(3, Suit.NT, valid)
            if b:
                return b

        return PASS

    # ------------------------------------------------------------------
    # OVERCALL
    # ------------------------------------------------------------------

    def _bid_overcall(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        h = hcp(hand)
        shape = hand_shape(hand)

        # 1NT overcall: 15-17 balanced with stopper in their suit
        calls = obs.get('calls', [])
        opp_suit = None
        for c in calls:
            if not c.special:
                opp_suit = c.strain
                break

        if self.params.overcall_1nt_min <= h <= self.params.overcall_1nt_max and shape.is_balanced and opp_suit and stopper(hand, opp_suit):
            b = self._best_valid(1, Suit.NT, valid)
            if b:
                return b

        # Suit overcall: 10-16 HCP with 5+ card suit and quality
        if self.params.overcall_min_hcp <= h <= self.params.overcall_max_hcp:
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if shape.length(suit) >= 5 and suit_quality(hand, suit) >= 2:
                    b = self._cheapest_in_suit(suit, valid)
                    if b:
                        return b

        # Strong hand: double or bid
        if h >= self.params.overcall_strong_min:
            if DOUBLE in valid:
                return DOUBLE
            # bid own suit
            for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
                if shape.length(suit) >= 5:
                    b = self._cheapest_in_suit(suit, valid)
                    if b:
                        return b

        return PASS

    # ------------------------------------------------------------------
    # COMPETITIVE
    # ------------------------------------------------------------------

    def _bid_competitive(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        h = hcp(hand)
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        my_bids = self._my_bids(calls, dealer)
        partner_bids = self._partner_bids(calls, dealer)
        fit = self._detected_fit(my_bids, partner_bids, hand)

        if fit:
            tp = total_points(hand, fit)
            est = self._estimate_partner(partner_bids, calls, dealer)
            partner_est = est.min_hcp + int((est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
            combined = tp + partner_est
            target_lv, target_st = self._target_level(min(combined, 32), fit, hand)
            b = self._best_valid(target_lv, target_st, valid)
            if b:
                return b

        # Penalty double: only with strong trump holding in their suit
        if h >= self.params.competitive_double_min and DOUBLE in valid:
            opp_bids = self._opp_bids(calls, dealer)
            if opp_bids:
                opp_suit = opp_bids[-1].strain
                opp_trump_holding = suit_length(hand, opp_suit)
                # Only double if we have 4+ of their trump suit (trump stack)
                if opp_trump_holding >= self.params.competitive_double_trump_len and suit_quality(hand, opp_suit) >= 2:
                    return DOUBLE

        return PASS

    # ------------------------------------------------------------------
    # SLAM INVESTIGATION
    # ------------------------------------------------------------------

    def _bid_slam(self, obs: dict) -> Bid:
        hand = obs['hand']
        valid = obs['valid_calls']
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        my_bids = self._my_bids(calls, dealer)
        partner_bids = self._partner_bids(calls, dealer)
        fit = self._detected_fit(my_bids, partner_bids, hand)
        est = self._estimate_partner(partner_bids, calls, dealer)
        tp = total_points(hand, fit)
        # Conservative: use min + 1/3 of range
        partner_est = est.min_hcp + int((est.max_hcp - est.min_hcp) * self.params.partner_est_fraction)
        combined = tp + partner_est

        # Check if we already bid 4NT (Blackwood) — partner should respond
        my_calls_all = [calls[i] for i in range(len(calls))
                        if (dealer + i) % 4 == self.seat]
        partner_calls_all = self._partner_calls(calls, dealer)
        i_asked_blackwood = any(
            c.level == 4 and c.strain == Suit.NT and not c.special
            for c in my_calls_all)
        partner_asked_blackwood = any(
            c.level == 4 and c.strain == Suit.NT and not c.special
            for c in partner_calls_all)

        # If partner responded to our Blackwood
        if i_asked_blackwood and partner_bids:
            last_partner = partner_calls_all[-1] if partner_calls_all else None
            if last_partner and last_partner.level == 5 and not last_partner.special:
                # Count aces from response
                partner_aces = {Suit.C: 0, Suit.D: 1, Suit.H: 2,
                                Suit.S: 3, Suit.NT: 4}.get(
                    last_partner.strain, 0)
                my_aces = sum(1 for c in hand if c.rank == Rank.ACE)
                total_aces = partner_aces + my_aces

                strain = fit if fit else Suit.NT
                # Grand slam: need all 4 aces
                if combined >= self.params.slam_grand_min and total_aces == 4:
                    b = self._best_valid(7, strain, valid)
                    if b:
                        return b
                # Small slam: need at least 3 aces
                if total_aces >= 3:
                    b = self._best_valid(6, strain, valid)
                    if b:
                        return b
                # missing aces → sign off at 5
                b = self._best_valid(5, strain, valid)
                if b:
                    return b

        # If partner asked Blackwood, respond with ace count
        if partner_asked_blackwood:
            my_aces = sum(1 for c in hand if c.rank == Rank.ACE)
            responses = {0: Suit.C, 1: Suit.D, 2: Suit.H, 3: Suit.S, 4: Suit.NT}
            resp_strain = responses[my_aces]
            b = self._best_valid(5, resp_strain, valid)
            if b:
                return b

        # Only investigate slam with a fit and enough quick tricks
        my_qt = quick_tricks(hand)
        if combined >= self.params.slam_small_min and fit and my_qt >= self.params.slam_min_qt:
            # Initiate Blackwood
            b = self._best_valid(4, Suit.NT, valid)
            if b:
                return b

        # Direct slam only with extreme strength + fit
        if combined >= self.params.slam_grand_min and fit and my_qt >= self.params.slam_direct_min_qt:
            b = self._best_valid(6, fit, valid)
            if b:
                return b

        # Fallback: bid game (not slam)
        target_lv, target_st = self._target_level(min(combined, 32), fit, hand)
        b = self._best_valid(target_lv, target_st, valid)
        return b if b else PASS

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def bid(self, obs: dict) -> Bid:
        """Select a bid given the current observation.

        Args:
            obs: Observation dict from GameState.observation().

        Returns:
            A legal Bid from obs['valid_calls'].
        """
        valid = obs['valid_calls']
        if len(valid) == 1:
            return valid[0]

        # Stash calls for sub-methods that need opponent suit info
        self._current_obs_calls = obs.get('calls', [])
        self._current_dealer = obs.get('dealer', 0)

        # Apply vulnerability adjustments
        vul_info = obs.get('vulnerable', {})
        my_side = 'NS' if self.seat % 2 == 0 else 'EW'
        my_vul = vul_info.get(my_side, False)
        self._effective_open_min = self.params.open_min_hcp
        self._effective_game_min = self.params.game_combined_min
        if not my_vul:
            # Not vulnerable: open lighter, games cheaper to bid
            self._effective_open_min -= self.params.vul_open_adjust
            self._effective_game_min -= self.params.vul_game_adjust
        else:
            # Vulnerable: tighter — penalties hurt more
            self._effective_open_min += self.params.vul_open_adjust
            self._effective_game_min += self.params.vul_game_adjust

        phase = self._determine_phase(obs)

        if phase == BidPhase.OPENING:
            result = self._bid_opening(obs)
        elif phase == BidPhase.RESPONDING:
            result = self._bid_responding(obs)
        elif phase == BidPhase.OPENER_REBID:
            result = self._bid_opener_rebid(obs)
        elif phase == BidPhase.RESPONDER_REBID:
            result = self._bid_responder_rebid(obs)
        elif phase == BidPhase.OVERCALL:
            result = self._bid_overcall(obs)
        elif phase == BidPhase.COMPETITIVE:
            result = self._bid_competitive(obs)
        elif phase == BidPhase.SLAM_INVESTIGATION:
            result = self._bid_slam(obs)
        else:
            result = PASS

        # Safety: ensure result is valid
        if result not in valid:
            result = PASS if PASS in valid else valid[0]

        # Emit trace if enabled
        if self.params.trace_enabled:
            hand = obs.get('hand', [])
            calls = obs.get('calls', [])
            dealer = obs.get('dealer', 0)
            h = hcp(hand)
            p_bids = self._partner_bids(calls, dealer)
            est = self._estimate_partner(p_bids, calls, dealer)
            partner_hcp = est.min_hcp + (est.max_hcp - est.min_hcp) * self.params.partner_est_fraction
            m_bids = self._my_bids(calls, dealer)
            fit = self._detected_fit(m_bids, p_bids, hand)
            self.last_trace = DecisionTrace(
                action_type='bid',
                seat=self.seat,
                phase=phase.name,
                chosen=str(result),
                reason=phase.name.lower(),
                details={
                    'hcp': h,
                    'partner_est': round(partner_hcp, 1),
                    'combined': round(h + partner_hcp, 1),
                    'fit': str(fit) if fit else None,
                },
            )

        return result
