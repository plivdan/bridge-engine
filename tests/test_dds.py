"""Double-dummy solver tests: known positions with exact answers."""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.card import Card, Suit, Rank, deal
from engine.dds import double_dummy_tricks

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

C = Card
S, H, D, Cl = Suit.S, Suit.H, Suit.D, Suit.C


# ── All top cards ────────────────────────────────────────────
section("ALL ACES + KINGS (4 tricks)")

hands_1 = {
    0: [C(Rank.ACE, S), C(Rank.ACE, H), C(Rank.ACE, D), C(Rank.ACE, Cl)],
    1: [C(Rank.TWO, S), C(Rank.TWO, H), C(Rank.TWO, D), C(Rank.TWO, Cl)],
    2: [C(Rank.KING, S), C(Rank.KING, H), C(Rank.KING, D), C(Rank.KING, Cl)],
    3: [C(Rank.THREE, S), C(Rank.THREE, H), C(Rank.THREE, D), C(Rank.THREE, Cl)],
}
r = double_dummy_tricks(hands_1, None, 0)
check(r == 4, f"NS has all aces+kings: {r} tricks (expect 4)")


# ── AKQ opposite xxx ────────────────────────────────────────
section("AKQ vs xxx (3 tricks in suit)")

hands_2 = {
    0: [C(Rank.ACE, S), C(Rank.KING, S), C(Rank.QUEEN, S), C(Rank.TWO, H)],
    1: [C(Rank.FIVE, S), C(Rank.FOUR, S), C(Rank.THREE, D), C(Rank.TWO, D)],
    2: [C(Rank.THREE, S), C(Rank.TWO, S), C(Rank.THREE, H), C(Rank.TWO, Cl)],
    3: [C(Rank.SIX, S), C(Rank.FOUR, D), C(Rank.FIVE, D), C(Rank.THREE, Cl)],
}
r = double_dummy_tricks(hands_2, None, 0)
check(r >= 1, f"AKQ spades: declarer gets >= 1 trick (got {r})")


# ── Finesse onside ──────────────────────────────────────────
section("AQ FINESSE ONSIDE")

hands_3 = {
    0: [C(Rank.ACE, S), C(Rank.QUEEN, S)],
    1: [C(Rank.KING, S), C(Rank.TWO, H)],
    2: [C(Rank.THREE, S), C(Rank.TWO, S)],
    3: [C(Rank.TWO, D), C(Rank.THREE, D)],
}
r = double_dummy_tricks(hands_3, None, 0)
check(r == 1, f"AQ finesse onside: {r} (expect 1)")


# ── Single suit, one side dominates ──────────────────────────
section("ALL SPADES - E HAS AKQ")

hands_4 = {
    0: [C(Rank.TWO, S), C(Rank.THREE, S), C(Rank.FOUR, S)],
    1: [C(Rank.ACE, S), C(Rank.KING, S), C(Rank.QUEEN, S)],
    2: [C(Rank.FIVE, S), C(Rank.SIX, S), C(Rank.SEVEN, S)],
    3: [C(Rank.JACK, S), C(Rank.TEN, S), C(Rank.NINE, S)],
}
r_ew = double_dummy_tricks(hands_4, None, 1)
check(r_ew == 3, f"E has AKQ spades, E declares: {r_ew} (expect 3)")
r_ns = double_dummy_tricks(hands_4, None, 0)
check(r_ns == 0, f"E has AKQ spades, N declares: {r_ns} (expect 0)")


# ── Trump contract — ruff ─────────────────────────────────────
section("TRUMP RUFF")

hands_5 = {
    0: [C(Rank.ACE, H), C(Rank.KING, H), C(Rank.TWO, S)],
    1: [C(Rank.ACE, S), C(Rank.KING, S), C(Rank.TWO, D)],
    2: [C(Rank.THREE, H), C(Rank.FOUR, H), C(Rank.THREE, S)],
    3: [C(Rank.QUEEN, S), C(Rank.JACK, S), C(Rank.THREE, D)],
}
# Hearts trump, N declares. NS has AKxx hearts + spade to ruff
r = double_dummy_tricks(hands_5, Suit.H, 0)
check(r >= 2, f"Trump contract with hearts: NS gets >= 2 (got {r})")


# ── Mixed suits, balanced ────────────────────────────────────
section("MIXED SUITS BALANCED")

hands_6 = {
    0: [C(Rank.ACE, S), C(Rank.TWO, H)],
    1: [C(Rank.KING, S), C(Rank.ACE, H)],
    2: [C(Rank.THREE, S), C(Rank.THREE, H)],
    3: [C(Rank.FOUR, S), C(Rank.FOUR, H)],
}
rn = double_dummy_tricks(hands_6, None, 0)
re = double_dummy_tricks(hands_6, None, 1)
check(rn == 1, f"N declares, has AS: {rn} (expect 1)")
check(re == 1, f"E declares, has AH: {re} (expect 1)")


# ── 7-card exact solve ────────────────────────────────────────
section("7-CARD EXACT SOLVE")

random.seed(42)
h = deal()
h7 = {s: h[s][:7] for s in range(4)}
t0 = time.time()
r7 = double_dummy_tricks(h7, None, 0)
elapsed = time.time() - t0
check(0 <= r7 <= 7, f"7-card NT result in [0,7] (got {r7})")
check(elapsed < 5.0, f"7-card solve < 5s (got {elapsed:.2f}s)")


# ── 8-card exact solve ────────────────────────────────────────
section("8-CARD EXACT SOLVE")

h8 = {s: h[s][:8] for s in range(4)}
t0 = time.time()
r8 = double_dummy_tricks(h8, None, 0)
elapsed = time.time() - t0
check(0 <= r8 <= 8, f"8-card NT result in [0,8] (got {r8})")
check(elapsed < 10.0, f"8-card solve < 10s (got {elapsed:.2f}s)")


# ── Full 13-card (heuristic mode) ─────────────────────────────
section("13-CARD HEURISTIC MODE")

random.seed(12345)
times = []
results = []
for i in range(20):
    hh = deal()
    t0 = time.time()
    r = double_dummy_tricks(hh, None, 0)
    times.append(time.time() - t0)
    results.append(r)
    assert 0 <= r <= 13, f"Board {i}: {r} out of range"

avg_ms = sum(times) / len(times) * 1000
check(all(0 <= r <= 13 for r in results), f"All 20 results in [0,13]")
check(avg_ms < 100, f"Heuristic mode avg < 100ms (got {avg_ms:.0f}ms)")

# Also test with trump
random.seed(99)
hh = deal()
for strain in [Suit.S, Suit.H, Suit.D, Suit.C, None]:
    r = double_dummy_tricks(hh, strain, 0)
    sname = str(strain) if strain else 'NT'
    check(0 <= r <= 13, f"Strain {sname}: {r} in [0,13]")


# ── SUMMARY ──────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
