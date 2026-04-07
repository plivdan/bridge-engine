"""Measurement and reporting: compare AI performance before/after changes.

Runs head-to-head matches and DD benchmarks, producing a summary
report with per-category breakdowns.
"""

import os
import sys
import random
import contextlib
import io
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai.bridge_params import BridgeParams
from ai.smart_player import SmartPlayer
from engine.player import RuleBasedPlayer
from engine.game import Game
from empirical.dd_benchmark import benchmark_vs_dd


def head_to_head(params: BridgeParams, num_boards: int = 1000,
                 seed: int = 42, label: str = '') -> dict:
    """Run SmartPlayer(params) NS vs RuleBasedPlayer EW.

    Returns dict with score summary.
    """
    random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        results = Game(
            [SmartPlayer(0, params), RuleBasedPlayer(1),
             SmartPlayer(2, params), RuleBasedPlayer(3)],
            num_boards=num_boards
        ).run()

    ns_total = sum(r['score_ns'] for r in results)
    ew_total = sum(r['score_ew'] for r in results)
    net = ns_total - ew_total
    ns_wins = sum(1 for r in results if r['score_ns'] > r['score_ew'])
    ew_wins = sum(1 for r in results if r['score_ew'] > r['score_ns'])

    return {
        'label': label,
        'num_boards': num_boards,
        'ns_total': ns_total,
        'ew_total': ew_total,
        'net': net,
        'per_board': net / num_boards,
        'ns_wins': ns_wins,
        'ew_wins': ew_wins,
        'win_pct': ns_wins / num_boards * 100,
    }


def full_report(num_boards: int = 500, seed: int = 42):
    """Generate a full comparison report: default vs optimized."""
    print('=' * 60)
    print('BRIDGE AI PERFORMANCE REPORT')
    print('=' * 60)

    t0 = time.time()

    # Default params
    print('\n--- Head-to-Head: Default Params ---')
    default_result = head_to_head(
        BridgeParams(), num_boards=num_boards, seed=seed,
        label='Default'
    )
    print(f"  Net advantage: {default_result['net']:+d} "
          f"({default_result['per_board']:+.1f}/board)")
    print(f"  NS wins: {default_result['ns_wins']}/{num_boards} "
          f"({default_result['win_pct']:.1f}%)")

    # Optimized params (if available)
    opt_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params_optimized.json')
    if os.path.exists(opt_path):
        opt_params = BridgeParams.from_json(opt_path)
        print('\n--- Head-to-Head: Optimized Params ---')
        opt_result = head_to_head(
            opt_params, num_boards=num_boards, seed=seed,
            label='Optimized'
        )
        print(f"  Net advantage: {opt_result['net']:+d} "
              f"({opt_result['per_board']:+.1f}/board)")
        print(f"  NS wins: {opt_result['ns_wins']}/{num_boards} "
              f"({opt_result['win_pct']:.1f}%)")

        # Improvement
        improvement = opt_result['net'] - default_result['net']
        print(f"\n  Improvement: {improvement:+d} total "
              f"({improvement / num_boards:+.1f}/board)")
    else:
        print('\n  (No optimized params found — run bridge_optimize.py first)')

    # DD Benchmark
    print('\n--- DD Benchmark (default params, 100 boards) ---')
    dd_result = benchmark_vs_dd(
        player_factory=lambda: [SmartPlayer(i) for i in range(4)],
        num_boards=100, seed=seed
    )
    print(f"  Mean AI tricks: {dd_result.mean_ai_tricks:.2f}")
    print(f"  Mean DD tricks: {dd_result.mean_dd_tricks:.2f}")
    print(f"  Mean trick gap: {dd_result.mean_trick_gap:.2f}")
    print(f"  Optimal play:   {dd_result.pct_optimal:.1f}%")

    # Worst boards
    print('\n  Worst 5 boards (largest trick gap):')
    for b in dd_result.worst_boards(5):
        print(f"    Board {b.board_num:3d}: AI={b.ai_tricks} DD={b.dd_tricks} "
              f"gap={b.trick_gap:+d} level={b.contract_level}")

    print(f'\n  Total report time: {time.time() - t0:.0f}s')
    print('=' * 60)


if __name__ == '__main__':
    full_report()
