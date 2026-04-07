"""Bidding agent correctness tests.

Tests crafted hands against expected bids and full auction outcomes.
Uses the same check/section framework as test_bridge.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.card import Card, Suit, Rank
from engine.auction import AuctionState, Bid, PASS, DOUBLE, REDOUBLE, make_bid
from ai.hand_eval import hcp, hand_shape, total_points, distribution_points
from ai.bidding_agent import StateMachineBidder

PASS_COUNT = 0
FAIL_COUNT = 0

def check(cond, label):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  PASS  {label}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {label}")

def section(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


# --- helpers ---------------------------------------------------------------

def make_obs(hand, calls=None, dealer=0, player=0, vulnerable=None):
    """Build a minimal observation dict for bidding tests."""
    calls = calls or []
    a = AuctionState(dealer=dealer)
    for c in calls:
        a.apply_call(c)
    return {
        'hand': hand,
        'calls': calls,
        'dealer': dealer,
        'player': player,
        'valid_calls': a.valid_calls(),
        'contract': a.contract,
        'declarer': a.declarer,
        'vulnerable': vulnerable or {'NS': False, 'EW': False},
    }


def bid_of(bidder, obs):
    """Get bid from a StateMachineBidder."""
    return bidder.bid(obs)


# --- hands -----------------------------------------------------------------

# 17 HCP balanced, all suits stopped → 1NT
HAND_1NT = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]

# 13 HCP + 5 spades → 1S
HAND_1S = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.NINE, Suit.S), Card(Rank.SEVEN, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]

# 20 HCP balanced (4-3-3-3) → 2NT
HAND_2NT = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.THREE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.EIGHT, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C),
]

# 22 HCP balanced (4-3-3-3) → 2C (artificial strong)
HAND_2C = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.QUEEN, Suit.H), Card(Rank.EIGHT, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.JACK, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.QUEEN, Suit.C), Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C),
]

# 6 HCP, weak → pass
HAND_WEAK = [
    Card(Rank.KING, Suit.S), Card(Rank.SEVEN, Suit.S), Card(Rank.THREE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.FIVE, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.NINE, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.FOUR, Suit.C), Card(Rank.TWO, Suit.C),
]

# 13 HCP, 5 hearts → 1H
HAND_1H = [
    Card(Rank.QUEEN, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H), Card(Rank.NINE, Suit.H),
    Card(Rank.SEVEN, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FIVE, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]

# 12 HCP, 4-4 minors, no 5-card major → 1D (longer minor)
HAND_1D = [
    Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]

# Responding hand: 10 HCP, 4 hearts (for limit raise after partner opens 1H)
HAND_RESP_HEARTS = [
    Card(Rank.QUEEN, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.NINE, Suit.H),
    Card(Rank.SEVEN, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.SIX, Suit.C), Card(Rank.FOUR, Suit.C), Card(Rank.TWO, Suit.C),
]

# Responding hand: 10 HCP, balanced → should bid 3NT over partner's 1NT
HAND_RESP_NT = [
    Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.TWO, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.FIVE, Suit.C),
]

# Weak responding hand: 4 HCP → should pass
HAND_RESP_WEAK = [
    Card(Rank.SEVEN, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.NINE, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.SIX, Suit.H),
    Card(Rank.QUEEN, Suit.D), Card(Rank.FOUR, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]


# ── OPENING BID SELECTION ────────────────────────────────────────
section("OPENING BID SELECTION")

b = StateMachineBidder(0)

obs = make_obs(HAND_1NT)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.NT), f"15 HCP balanced -> 1NT (got {bid})")

obs = make_obs(HAND_1S)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.S), f"13 HCP + 5S -> 1S (got {bid})")

obs = make_obs(HAND_2NT)
bid = b.bid(obs)
check(bid == make_bid(2, Suit.NT), f"20 HCP balanced -> 2NT (got {bid})")

obs = make_obs(HAND_2C)
bid = b.bid(obs)
check(bid == make_bid(2, Suit.C), f"22+ HCP -> 2C (got {bid})")

obs = make_obs(HAND_WEAK)
bid = b.bid(obs)
check(bid == PASS, f"8 HCP weak -> PASS (got {bid})")

obs = make_obs(HAND_1H)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.H), f"13 HCP + 5H -> 1H (got {bid})")

obs = make_obs(HAND_1D)
bid = b.bid(obs)
check(bid.strain in (Suit.D, Suit.C) and bid.level == 1,
      f"12 HCP minors -> 1m (got {bid})")


# ── RESPONDING ────────────────────────────────────────────────────
section("RESPONDING")

# Partner (seat 0) opens 1H, we are seat 2 responding
# N opens 1H, E passes, S responds
resp_bidder = StateMachineBidder(2)

calls_1h = [make_bid(1, Suit.H), PASS]  # N:1H, E:PASS, now S bids
obs = make_obs(HAND_RESP_HEARTS, calls=calls_1h, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid.strain == Suit.H and bid.level >= 2,
      f"10 HCP + 4H, partner opens 1H -> raise hearts (got {bid})")
check(bid.level == 3,
      f"10 HCP + 4H should be limit raise to 3H (got {bid})")

# Partner opens 1NT, we have 10 HCP → 3NT
calls_1nt = [make_bid(1, Suit.NT), PASS]
obs = make_obs(HAND_RESP_NT, calls=calls_1nt, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == make_bid(3, Suit.NT), f"10 HCP over 1NT -> 3NT (got {bid})")

# Partner opens 1H, we have 4 HCP → PASS
calls_1h2 = [make_bid(1, Suit.H), PASS]
obs = make_obs(HAND_RESP_WEAK, calls=calls_1h2, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == PASS, f"4 HCP over 1H -> PASS (got {bid})")

# Partner opens 1NT, we have 4 HCP → PASS
calls_1nt2 = [make_bid(1, Suit.NT), PASS]
obs = make_obs(HAND_RESP_WEAK, calls=calls_1nt2, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == PASS, f"4 HCP over 1NT -> PASS (got {bid})")


# ── FULL AUCTION: GAME REACHED ───────────────────────────────────
section("FULL AUCTION: GAME REACHED")

import contextlib, io

def run_auction(hands, dealer=0):
    """Run a full auction with SmartBidders and return the result."""
    from engine.state import GameState
    gs = GameState(board_num=1, vulnerable={'NS': False, 'EW': False}, dealer=dealer)
    gs.hands = hands
    gs.auction = AuctionState(dealer=dealer)
    gs.phase = 'AUCTION'
    bidders = {i: StateMachineBidder(i) for i in range(4)}
    with contextlib.redirect_stdout(io.StringIO()):
        while gs.phase == 'AUCTION':
            actor = gs.next_actor()
            obs = gs.observation(actor)
            call = bidders[actor].bid(obs)
            gs.apply_call(call)
    return gs.auction

# NS have 26+ combined points with 8-spade fit → should reach 4S
hand_n_game = [  # 14 HCP + 5S
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.NINE, Suit.S), Card(Rank.SEVEN, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]
hand_s_game = [  # 13 HCP + 3S support
    Card(Rank.QUEEN, Suit.S), Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SEVEN, Suit.C),
    Card(Rank.TWO, Suit.C),
]
hand_e_weak = [  # 5 HCP
    Card(Rank.EIGHT, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.NINE, Suit.H), Card(Rank.SEVEN, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.TWO, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.TEN, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.FOUR, Suit.C), Card(Rank.THREE, Suit.C),
]
hand_w_weak = [  # 5 HCP
    Card(Rank.SIX, Suit.S), Card(Rank.THREE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.TEN, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.SIX, Suit.H),
    Card(Rank.TEN, Suit.D), Card(Rank.NINE, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.SIX, Suit.D), Card(Rank.FIVE, Suit.D),
    Card(Rank.ACE, Suit.C), Card(Rank.EIGHT, Suit.C),
]

auction = run_auction({0: hand_n_game, 1: hand_e_weak, 2: hand_s_game, 3: hand_w_weak})
contract = auction.contract
declarer = auction.declarer
check(contract is not None, "game auction produced a contract")
if contract:
    check(contract.level >= 3,
          f"NS with 27 pts reached level {contract.level} (need 3+)")
    check(declarer in (0, 2),
          f"NS should be declarer (got seat {declarer})")


# NS have ~24 combined → should stay below game
hand_n_part = [  # 12 HCP
    Card(Rank.ACE, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.NINE, Suit.S),
    Card(Rank.SEVEN, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.QUEEN, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.FOUR, Suit.C),
    Card(Rank.TWO, Suit.C),
]
hand_s_part = [  # 10 HCP
    Card(Rank.KING, Suit.S), Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.NINE, Suit.C), Card(Rank.SEVEN, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.THREE, Suit.C),
]

auction2 = run_auction({0: hand_n_part, 1: hand_e_weak, 2: hand_s_part, 3: hand_w_weak})
contract2 = auction2.contract
if contract2:
    check(contract2.level <= 3,
          f"NS with ~22 pts should stay low (got {contract2.level}{contract2.strain})")
else:
    check(True, "NS with ~22 pts: passed out is acceptable")


# ── HAND EVALUATION CONSISTENCY ──────────────────────────────────
section("HAND EVALUATION CONSISTENCY")

check(hcp(HAND_1NT) == 17, f"HAND_1NT HCP = {hcp(HAND_1NT)} (expect 17)")
check(hcp(HAND_1S) == 13, f"HAND_1S HCP = {hcp(HAND_1S)} (expect 13)")
check(hcp(HAND_2NT) == 20, f"HAND_2NT HCP = {hcp(HAND_2NT)} (expect 20)")
check(hcp(HAND_2C) == 22, f"HAND_2C HCP = {hcp(HAND_2C)} (expect 22)")
check(hcp(HAND_WEAK) == 6, f"HAND_WEAK HCP = {hcp(HAND_WEAK)} (expect 6)")
check(hand_shape(HAND_1NT).is_balanced, "HAND_1NT is balanced")
check(hand_shape(HAND_1H).is_balanced,
      "HAND_1H 5-3-3-2 is balanced")
check(hand_shape(HAND_1H).longest_suit == Suit.H,
      "HAND_1H longest suit is hearts")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
