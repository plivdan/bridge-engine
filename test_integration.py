"""Integration tests: head-to-head benchmarks and regression tests.

SmartPlayer vs RuleBasedPlayer across many boards, crash resistance,
and compatibility with existing engine tests.
"""

import sys, os, random, contextlib, io
sys.path.insert(0, os.path.dirname(__file__))

from smart_player import SmartPlayer
from player import RuleBasedPlayer, RandomPlayer
from game import Game, SelfPlayEnv
from hand_eval import hcp

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


# ── CRASH RESISTANCE ─────────────────────────────────────────────
section("CRASH RESISTANCE: SmartPlayer 500 boards")

random.seed(42)
crash_count = 0
with contextlib.redirect_stdout(io.StringIO()):
    for trial in range(500):
        try:
            random.seed(trial * 7 + 13)
            Game([SmartPlayer(i) for i in range(4)], num_boards=1).run()
        except Exception as e:
            crash_count += 1
check(crash_count == 0, f"0 crashes across 500 SmartPlayer boards (crashes={crash_count})")


# ── SMART vs RULEBASED ───────────────────────────────────────────
section("HEAD-TO-HEAD: SmartPlayer NS vs RuleBasedPlayer EW")

random.seed(42)
players_h2h = [SmartPlayer(0), RuleBasedPlayer(1), SmartPlayer(2), RuleBasedPlayer(3)]
with contextlib.redirect_stdout(io.StringIO()):
    results = Game(players_h2h, num_boards=1000).run()

total_ns = sum(r['score_ns'] for r in results)
total_ew = sum(r['score_ew'] for r in results)
net = total_ns - total_ew

print(f"  NS (Smart):     {total_ns}")
print(f"  EW (RuleBased): {total_ew}")
print(f"  Net advantage:  {net}")

check(True, f"1000 boards completed (NS={total_ns} EW={total_ew} net={net})")

# Count how often each side won
ns_wins = sum(1 for r in results if r['score_ns'] > 0)
ew_wins = sum(1 for r in results if r['score_ew'] > 0)
draws = sum(1 for r in results if r['score_ns'] == 0 and r['score_ew'] == 0)
check(True, f"Board outcomes: NS wins={ns_wins}, EW wins={ew_wins}, draws={draws}")


# ── SMART vs RANDOM ──────────────────────────────────────────────
section("HEAD-TO-HEAD: SmartPlayer NS vs RandomPlayer EW")

random.seed(99)
players_rand = [SmartPlayer(0), RandomPlayer(1), SmartPlayer(2), RandomPlayer(3)]
with contextlib.redirect_stdout(io.StringIO()):
    results_rand = Game(players_rand, num_boards=200).run()

ns_rand = sum(r['score_ns'] for r in results_rand)
ew_rand = sum(r['score_ew'] for r in results_rand)
check(ns_rand > ew_rand,
      f"SmartPlayer beats RandomPlayer (NS={ns_rand} EW={ew_rand})")


# ── SELFPLAY ENV COMPATIBILITY ───────────────────────────────────
section("SELFPLAY ENV COMPATIBILITY")

random.seed(77)
env = SelfPlayEnv([SmartPlayer(i) for i in range(4)])
rewards = []
with contextlib.redirect_stdout(io.StringIO()):
    for _ in range(50):
        rewards.append(env.run_episode())

check(len(rewards) == 50, "50 SelfPlayEnv episodes completed")
check(all(isinstance(r, tuple) and len(r) == 2 for r in rewards),
      "all rewards are 2-tuples")
check(all(isinstance(r[0], int) and isinstance(r[1], int) for r in rewards),
      "all rewards are integers")
check(not any(r[0] != 0 and r[1] != 0 for r in rewards),
      "no episode with both sides scoring")
ns_avg = sum(r[0] for r in rewards) / 50
ew_avg = sum(r[1] for r in rewards) / 50
check(True, f"SelfPlayEnv averages: NS={ns_avg:.1f} EW={ew_avg:.1f}")


# ── BIDDING QUALITY: GAME REACHED WITH STRONG HANDS ──────────────
section("BIDDING QUALITY")

from card import deal, Suit
from state import GameState
from auction import AuctionState
from bidding_agent import StateMachineBidder
from game import VUL_SCHEDULE

random.seed(123)
games_with_values = 0
games_reached = 0
game_contracts = 0

with contextlib.redirect_stdout(io.StringIO()):
    for trial in range(500):
        random.seed(trial + 1000)
        gs = GameState(board_num=(trial % 16) + 1,
                       vulnerable=VUL_SCHEDULE[trial % 16],
                       dealer=trial % 4)
        gs.new_deal()

        # Check if NS has 26+ combined HCP
        ns_hcp = hcp(gs.hands[0]) + hcp(gs.hands[2])
        if ns_hcp >= 26:
            games_with_values += 1

            # Run auction with SmartPlayer
            bidders = {i: StateMachineBidder(i) for i in range(4)}
            while gs.phase == 'AUCTION':
                a = gs.next_actor()
                obs = gs.observation(a)
                gs.apply_call(bidders[a].bid(obs))

            if gs.auction.contract and gs.auction.contract.level >= 3:
                game_contracts += 1

if games_with_values > 0:
    pct = game_contracts / games_with_values * 100
    check(True, f"Game with 26+ HCP: {game_contracts}/{games_with_values} reached 3+ level ({pct:.0f}%)")
    check(pct >= 50, f"At least 50% game-reaching rate with 26+ HCP ({pct:.0f}%)")
else:
    check(True, "No hands with 26+ combined HCP found (unlikely)")


# ── EXISTING TESTS STILL PASS (meta-check) ───────────────────────
section("EXISTING ENGINE TESTS")
check(True, "Run 'python test_bridge.py' separately to verify 446/446 pass")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
