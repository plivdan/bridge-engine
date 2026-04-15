"""Auction-inference tests (Batch 9).

Verifies that infer_from_auction produces the HCP and shape constraints
any competent player would read from the bidding.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.card import Card, Suit, Rank
from engine.auction import Bid, PASS, DOUBLE, make_bid
from ai.inference import infer_from_auction, SeatConstraints
from ai.bridge_params import BridgeParams

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


# ── 1NT OPENING INFERENCE ────────────────────────────────────────
section("INFER FROM 1NT OPENING")

p = BridgeParams()
calls = [make_bid(1, Suit.NT), PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)

c0 = constraints[0]
check(c0.hcp_min == p.open_1nt_min,
      f"1NT opener min HCP = {c0.hcp_min} (expect {p.open_1nt_min})")
check(c0.hcp_max == p.open_1nt_max,
      f"1NT opener max HCP = {c0.hcp_max} (expect {p.open_1nt_max})")
check(c0.is_balanced is True, "1NT opener is balanced")
check(c0.suit_min[Suit.S] >= 2 and c0.suit_max[Suit.S] <= 5,
      f"1NT opener S length {c0.suit_min[Suit.S]}-{c0.suit_max[Suit.S]}")


# ── 1-OF-MAJOR OPENING ───────────────────────────────────────────
section("INFER FROM 1H OPENING")

calls = [make_bid(1, Suit.H), PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)
c0 = constraints[0]
check(c0.hcp_min == p.open_min_hcp,
      f"1H opener min HCP = {c0.hcp_min}")
check(c0.suit_min[Suit.H] >= 5, f"1H opener has 5+ hearts")


# ── WEAK 2 INFERENCE ─────────────────────────────────────────────
section("INFER FROM WEAK 2 OPENING")

calls = [make_bid(2, Suit.S), PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)
c0 = constraints[0]
check(c0.hcp_min == p.weak_two_min_hcp,
      f"2S weak min HCP = {c0.hcp_min}")
check(c0.hcp_max == p.weak_two_max_hcp,
      f"2S weak max HCP = {c0.hcp_max}")
check(c0.suit_min[Suit.S] == 6 and c0.suit_max[Suit.S] == 6,
      f"2S weak shows exactly 6 spades")


# ── PREEMPT INFERENCE ────────────────────────────────────────────
section("INFER FROM 3-LEVEL PREEMPT")

calls = [make_bid(3, Suit.C), PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)
c0 = constraints[0]
check(c0.suit_min[Suit.C] >= 7, f"3C preempt has 7+ clubs")
check(c0.hcp_max == p.preempt_3_max_hcp,
      f"3C preempt max HCP = {c0.hcp_max}")


# ── PASSED HAND INFERENCE ────────────────────────────────────────
section("INFER FROM PASS-OUT")

# All four seats pass — all four are passed hands.
calls = [PASS, PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)
for s in range(4):
    check(constraints[s].hcp_max < p.open_min_hcp,
          f"seat {s} passed out: max HCP = {constraints[s].hcp_max}")


# ── PARTNER-OPENS, ME-HAS-PASSED ─────────────────────────────────
section("INFER: I PASSED, PARTNER OPENED")

# Dealer=seat 0 passes; seat 1 passes; seat 2 opens 1NT; seat 3 passes.
calls = [PASS, PASS, make_bid(1, Suit.NT), PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)

c0 = constraints[0]
c2 = constraints[2]
check(c0.hcp_max < p.open_min_hcp,
      f"seat 0 passed before opening: max HCP = {c0.hcp_max}")
check(c2.hcp_min == p.open_1nt_min and c2.is_balanced is True,
      f"seat 2 opened 1NT: balanced {c2.hcp_min}-{c2.hcp_max}")


# ── HAND CONSISTENCY FILTER ──────────────────────────────────────
section("HAND CONSISTENCY: 1NT OPENER SAMPLE")

# A hand that looks like 1NT: 16 HCP balanced 4-3-3-3
hand_1nt = [
    Card(Rank.ACE, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.SEVEN, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.FIVE, Suit.H),
    Card(Rank.TWO, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.THREE, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.FOUR, Suit.C),
]

calls = [make_bid(1, Suit.NT), PASS, PASS, PASS]
constraints = infer_from_auction(calls, dealer=0, params=p)
check(constraints[0].hand_is_consistent(hand_1nt),
      "1NT-opener hand consistent with 1NT constraints")

# A hand that does NOT match 1NT: 22 HCP unbalanced
hand_big = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
    Card(Rank.QUEEN, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.TEN, Suit.S), Card(Rank.ACE, Suit.H),
    Card(Rank.KING, Suit.H), Card(Rank.QUEEN, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.KING, Suit.D),
    Card(Rank.ACE, Suit.C), Card(Rank.KING, Suit.C),
    Card(Rank.TWO, Suit.C),
]
check(not constraints[0].hand_is_consistent(hand_big),
      "22-HCP hand NOT consistent with 1NT opener")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
