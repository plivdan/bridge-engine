"""Empirical framework tests: params, data collection, tables, optimization."""

import sys, os, random, contextlib, io, json, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from bridge_params import BridgeParams
from bridge_stats import collect_boards, records_to_csv, records_from_csv, analyze_records
from bridge_tables import EmpiricalTables, expected_hcp_from_dist
from smart_player import SmartPlayer
from player import RuleBasedPlayer
from game import Game
from card import Card, Suit, Rank
from auction import make_bid, PASS, AuctionState
from bidding_agent import StateMachineBidder

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


# ── PARAMS: DEFAULTS REPRODUCE CURRENT BEHAVIOR ─────────────────
section("PARAMS: DEFAULTS AND JSON ROUND-TRIP")

p = BridgeParams()
check(p.game_combined_min == 26, "default game threshold = 26")
check(p.open_min_hcp == 12, "default open min = 12")
check(p.partner_est_fraction == 0.333, "default partner fraction = 0.333")
check(p.use_monte_carlo == False, "MC disabled by default")
check(p.slam_small_min == 33, "default slam threshold = 33")

# JSON round-trip
with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
    tmp_path = f.name
p.to_json(tmp_path)
p2 = BridgeParams.from_json(tmp_path)
check(p2.game_combined_min == 26, "JSON round-trip preserves game_combined_min")
check(p2.open_min_hcp == 12, "JSON round-trip preserves open_min_hcp")
check(p2.partner_est_fraction == 0.333, "JSON round-trip preserves partner_est_fraction")
check(p2.use_monte_carlo == False, "JSON round-trip preserves use_monte_carlo")
os.unlink(tmp_path)

# Custom params
p3 = BridgeParams(game_combined_min=24, open_min_hcp=11)
check(p3.game_combined_min == 24, "custom game threshold = 24")
check(p3.open_min_hcp == 11, "custom open min = 11")


# ── PARAMS: BIDDING REGRESSION ───────────────────────────────────
section("PARAMS: BIDDING REGRESSION (defaults = same bids)")

# Test that default params produce identical bids to the expected behavior
hand_1nt = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]
a = AuctionState(dealer=0)
obs = {'hand': hand_1nt, 'calls': [], 'dealer': 0, 'player': 0,
       'valid_calls': a.valid_calls(), 'contract': None, 'declarer': None,
       'vulnerable': {'NS': False, 'EW': False}}

bidder_default = StateMachineBidder(0)
bidder_params = StateMachineBidder(0, params=BridgeParams())
bid1 = bidder_default.bid(obs)
bid2 = bidder_params.bid(obs)
check(bid1 == bid2, f"default params bid matches: {bid1} == {bid2}")
check(bid1 == make_bid(1, Suit.NT), f"17 HCP balanced -> 1NT (got {bid1})")


# ── DATA COLLECTION ──────────────────────────────────────────────
section("DATA COLLECTION: collect_boards()")

records = collect_boards(
    player_factory=lambda: [SmartPlayer(i) for i in range(4)],
    num_boards=100, seed=42
)
check(len(records) == 100, f"collected 100 records (got {len(records)})")
check(all(r.board_num > 0 for r in records), "all board_nums > 0")
check(all(0 <= r.ns_combined_hcp <= 40 for r in records), "NS HCP in [0,40]")
check(all(r.ns_combined_hcp + r.ew_combined_hcp == 40 for r in records),
      "NS + EW HCP always = 40")
non_passed = [r for r in records if not r.passed_out]
check(all(1 <= r.contract_level <= 7 for r in non_passed), "contract levels in [1,7]")
check(all(r.tricks_ns + r.tricks_ew == 13 for r in non_passed), "tricks always sum to 13")

# CSV round-trip
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
    csv_path = f.name
records_to_csv(records, csv_path)
records2 = records_from_csv(csv_path)
check(len(records2) == 100, "CSV round-trip preserves count")
check(records2[0].board_num == records[0].board_num, "CSV preserves board_num")
check(records2[0].ns_combined_hcp == records[0].ns_combined_hcp, "CSV preserves HCP")
os.unlink(csv_path)

# Analyze
stats = analyze_records(records)
check(stats['total_boards'] == 100, f"analysis total = 100")
check('game_by_hcp' in stats, "analysis has game_by_hcp breakdown")


# ── EMPIRICAL TABLES ─────────────────────────────────────────────
section("EMPIRICAL TABLES")

# Build tables from collected data
tables = EmpiricalTables(records)

check(isinstance(tables.game_ev, dict), "game_ev is a dict")
check(isinstance(tables.slam_ev, dict), "slam_ev is a dict")
check(isinstance(tables.make_rates, dict), "make_rates is a dict")
check(isinstance(tables.partner_dists, dict), "partner_dists is a dict")

# Smoke test lookups
game_prof = tables.game_is_profitable(30, False)
check(isinstance(game_prof, bool), f"game_is_profitable returns bool (got {game_prof})")

slam_prof = tables.slam_is_profitable(35, True, False)
check(isinstance(slam_prof, bool), f"slam_is_profitable returns bool")

rate = tables.make_rate(26, 3, False)
check(0.0 <= rate <= 1.0, f"make_rate in [0,1] (got {rate:.2f})")

# Partner distributions
for cat, dist in tables.partner_dists.items():
    check(len(dist) == 41, f"partner dist '{cat}' has 41 elements")
    total = sum(dist)
    check(abs(total - 1.0) < 0.01, f"partner dist '{cat}' sums to ~1.0 (got {total:.3f})")
    e = expected_hcp_from_dist(dist)
    check(0 <= e <= 40, f"E[HCP|{cat}] = {e:.1f} in [0,40]")


# ── OPTIMIZED VS DEFAULT ─────────────────────────────────────────
section("OPTIMIZED VS DEFAULT BENCHMARK")

# Load optimized params if available
opt_path = os.path.join(os.path.dirname(__file__), 'params_optimized.json')
if os.path.exists(opt_path):
    opt_params = BridgeParams.from_json(opt_path)
    default_params = BridgeParams()

    random.seed(12345)
    with contextlib.redirect_stdout(io.StringIO()):
        r_default = Game(
            [SmartPlayer(0, default_params), RuleBasedPlayer(1),
             SmartPlayer(2, default_params), RuleBasedPlayer(3)],
            num_boards=500
        ).run()

    random.seed(12345)
    with contextlib.redirect_stdout(io.StringIO()):
        r_opt = Game(
            [SmartPlayer(0, opt_params), RuleBasedPlayer(1),
             SmartPlayer(2, opt_params), RuleBasedPlayer(3)],
            num_boards=500
        ).run()

    net_default = sum(r['score_ns'] - r['score_ew'] for r in r_default)
    net_opt = sum(r['score_ns'] - r['score_ew'] for r in r_opt)

    check(True, f"Default net: {net_default:+d}, Optimized net: {net_opt:+d}")
    check(net_opt >= net_default * 0.9,
          f"Optimized >= 90% of default ({net_opt:+d} vs {net_default:+d})")
else:
    check(True, "params_optimized.json not found (run bridge_optimize.py first)")


# ── MONTE CARLO SMOKE TEST ───────────────────────────────────────
section("MONTE CARLO SMOKE TEST")

mc_params = BridgeParams(use_monte_carlo=True, monte_carlo_samples=5, slam_small_min=99)
random.seed(42)
crash = 0
with contextlib.redirect_stdout(io.StringIO()):
    for t in range(20):
        try:
            random.seed(t + 9000)
            Game([SmartPlayer(i, params=mc_params) for i in range(4)], num_boards=1).run()
        except Exception as e:
            crash += 1
check(crash == 0, f"MC mode: 0 crashes in 20 boards (crashes={crash})")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
