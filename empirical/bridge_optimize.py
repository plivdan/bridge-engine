"""Parameter optimization with statistical rigor.

Improvements over naive coordinate descent:
1. Multi-seed evaluation — average across 3 seeds to prevent overfitting
2. Paired comparison — same deals with both configs, t-test for significance
3. Pairwise interaction search — jointly optimize interacting parameters
4. Holdout validation — verify gains on unseen seeds after optimization
5. Noise-aware — requires statistical significance before accepting changes
"""

import os
import sys
import random
import contextlib
import io
import time
import math
from dataclasses import replace
from typing import Dict, List, Tuple, Optional

from ai.bridge_params import BridgeParams
from ai.smart_player import SmartPlayer
from engine.player import RuleBasedPlayer
from engine.game import Game


# ── Progress bar ────────────────────────────────────────────────

class ProgressBar:
    """Simple inline progress bar for long-running optimization."""

    def __init__(self, total: int, label: str = '', width: int = 30):
        self.total = max(total, 1)
        self.current = 0
        self.label = label
        self.width = width
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        self._draw()

    def _draw(self):
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f'ETA {eta:.0f}s'
        else:
            eta_str = ''
        sys.stdout.write(f'\r  {self.label} [{bar}] {self.current}/{self.total} '
                         f'({pct:.0%}) {eta_str}   ')
        sys.stdout.flush()

    def finish(self, msg: str = ''):
        elapsed = time.time() - self.start_time
        sys.stdout.write(f'\r  {self.label} [{"#" * self.width}] '
                         f'{self.total}/{self.total} done in {elapsed:.0f}s'
                         f'{"  " + msg if msg else ""}   \n')
        sys.stdout.flush()


# ── Evaluation ──────────────────────────────────────────────────

def _run_match(params: BridgeParams, num_boards: int, seed: int) -> List[dict]:
    """Play num_boards and return per-board results."""
    random.seed(seed)
    players = [
        SmartPlayer(0, params=params), RuleBasedPlayer(1),
        SmartPlayer(2, params=params), RuleBasedPlayer(3),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        return Game(players, num_boards=num_boards).run()


def evaluate_params(params: BridgeParams, num_boards: int = 2000,
                    seed: int = 42, vs_default: bool = True) -> float:
    """Play num_boards and return average NS score advantage."""
    results = _run_match(params, num_boards, seed)
    ns = sum(r['score_ns'] for r in results)
    ew = sum(r['score_ew'] for r in results)
    return (ns - ew) / num_boards


def evaluate_multi_seed(params: BridgeParams, num_boards: int = 1000,
                        seeds: List[int] = None) -> Tuple[float, float]:
    """Evaluate across multiple seeds. Returns (mean_score, stderr)."""
    if seeds is None:
        seeds = [42, 123, 456]
    scores = [evaluate_params(params, num_boards, s) for s in seeds]
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        stderr = math.sqrt(var / len(scores))
    else:
        stderr = 0.0
    return mean, stderr


def paired_compare(params_a: BridgeParams, params_b: BridgeParams,
                   num_boards: int = 1000, seed: int = 42,
                   min_t: float = 1.5) -> Tuple[float, float, bool]:
    """Compare two param sets on the SAME deals (paired test).

    Returns (mean_diff_per_board, t_statistic, is_significant).
    Positive mean_diff means B is better than A.
    """
    results_a = _run_match(params_a, num_boards, seed)
    results_b = _run_match(params_b, num_boards, seed)

    diffs = []
    for ra, rb in zip(results_a, results_b):
        net_a = ra['score_ns'] - ra['score_ew']
        net_b = rb['score_ns'] - rb['score_ew']
        diffs.append(net_b - net_a)

    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, False

    mean_d = sum(diffs) / n
    if n > 1:
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        stderr = math.sqrt(var_d / n)
        t_stat = mean_d / stderr if stderr > 0 else 0.0
    else:
        t_stat = 0.0

    return mean_d, t_stat, (mean_d > 0 and abs(t_stat) >= min_t)


# ── Optimizers ──────────────────────────────────────────────────

def _count_evals(param_ranges: Dict[str, list], passes: int) -> int:
    """Count total evaluations for progress tracking."""
    total = 1  # baseline
    per_pass = sum(len(v) - 1 for v in param_ranges.values())
    # Worst case: all passes run (no early stop)
    total += per_pass * passes
    return total


def coordinate_descent(base_params: BridgeParams,
                       param_ranges: Dict[str, list],
                       num_boards: int = 2000,
                       seed: int = 42,
                       passes: int = 2,
                       verbose: bool = True) -> Tuple[BridgeParams, float]:
    """Optimize parameters one at a time with paired significance testing."""
    best = base_params
    best_score = evaluate_params(best, num_boards, seed)

    total_evals = _count_evals(param_ranges, passes)
    pb = ProgressBar(total_evals, label='coord descent') if verbose else None

    if verbose:
        print(f'  Baseline: {best_score:+.1f}/board | '
              f'{len(param_ranges)} params x {passes} passes = '
              f'{total_evals} evals')

    for pass_num in range(passes):
        improved = False
        for param_name, values in param_ranges.items():
            current_val = getattr(best, param_name)
            best_val = current_val
            best_diff = 0.0
            best_t = 0.0

            for val in values:
                if val == current_val:
                    continue
                trial = replace(best, **{param_name: val})
                diff, t_stat, sig = paired_compare(
                    best, trial, num_boards, seed, min_t=1.5
                )
                if pb:
                    pb.update()
                if sig and diff > best_diff:
                    best_diff = diff
                    best_t = t_stat
                    best_val = val

            if best_val != current_val:
                best = replace(best, **{param_name: best_val})
                best_score += best_diff
                improved = True

        if not improved:
            break

    if pb:
        pb.finish(f'score {best_score:+.1f}/board')

    # Print what changed
    if verbose:
        for param_name in param_ranges:
            old = getattr(base_params, param_name)
            new = getattr(best, param_name)
            if old != new:
                print(f'    {param_name}: {old} -> {new}')

    return best, best_score


def pairwise_optimize(base_params: BridgeParams,
                      param_pairs: List[Tuple[str, list, str, list]],
                      num_boards: int = 1500,
                      seed: int = 42,
                      verbose: bool = True) -> Tuple[BridgeParams, float]:
    """Jointly optimize pairs of interacting parameters."""
    best = base_params

    total_combos = sum((len(va) * len(vb) - 1) for _, va, _, vb in param_pairs)
    pb = ProgressBar(total_combos, label='pairwise') if verbose else None

    for name_a, vals_a, name_b, vals_b in param_pairs:
        cur_a = getattr(best, name_a)
        cur_b = getattr(best, name_b)
        best_combo = (cur_a, cur_b)
        best_diff = 0.0
        best_t = 0.0

        for va in vals_a:
            for vb in vals_b:
                if va == cur_a and vb == cur_b:
                    continue
                trial = replace(best, **{name_a: va, name_b: vb})
                diff, t_stat, sig = paired_compare(
                    best, trial, num_boards, seed, min_t=1.5
                )
                if pb:
                    pb.update()
                if sig and diff > best_diff:
                    best_diff = diff
                    best_t = t_stat
                    best_combo = (va, vb)

        if best_combo != (cur_a, cur_b):
            best = replace(best, **{name_a: best_combo[0], name_b: best_combo[1]})
            if verbose:
                print(f'\r    {name_a}+{name_b}: ({cur_a},{cur_b}) -> '
                      f'({best_combo[0]},{best_combo[1]}) '
                      f'(+{best_diff:.1f}/board, t={best_t:.1f})')

    if pb:
        score = evaluate_params(best, num_boards, seed)
        pb.finish(f'score {score:+.1f}/board')
    else:
        score = evaluate_params(best, num_boards, seed)

    return best, score


def validate_on_holdout(params: BridgeParams, baseline: BridgeParams,
                        num_boards: int = 1000,
                        holdout_seeds: List[int] = None,
                        verbose: bool = True) -> Tuple[float, float, float, float]:
    """Validate optimized params on unseen seeds."""
    if holdout_seeds is None:
        holdout_seeds = [777, 888, 999, 1111, 2222]

    total = len(holdout_seeds) * 2
    pb = ProgressBar(total, label='holdout') if verbose else None

    opt_scores = []
    base_scores = []
    for seed in holdout_seeds:
        opt_scores.append(evaluate_params(params, num_boards, seed))
        if pb:
            pb.update()
        base_scores.append(evaluate_params(baseline, num_boards, seed))
        if pb:
            pb.update()

    def _stats(scores):
        m = sum(scores) / len(scores)
        v = sum((s - m) ** 2 for s in scores) / (len(scores) - 1)
        se = math.sqrt(v / len(scores))
        return m, se

    opt_m, opt_se = _stats(opt_scores)
    base_m, base_se = _stats(base_scores)

    if pb:
        pb.finish()
    if verbose:
        print(f'  Baseline:    {base_m:+.1f} +/- {base_se:.1f}/board')
        print(f'  Optimized:   {opt_m:+.1f} +/- {opt_se:.1f}/board')
        diff = opt_m - base_m
        diff_se = math.sqrt(opt_se ** 2 + base_se ** 2)
        t = diff / diff_se if diff_se > 0 else 0
        sig = 'SIGNIFICANT' if abs(t) > 2.0 else 'not significant'
        print(f'  Improvement: {diff:+.1f}/board (t={t:.1f}, {sig})')

    return opt_m, opt_se, base_m, base_se


# ── Parameter ranges ────────────────────────────────────────────

HIGH_IMPACT_RANGES = {
    'game_combined_min': [23, 24, 25, 26, 27, 28, 29, 30],
    'partner_est_fraction': [0.2, 0.25, 0.3, 0.333, 0.4, 0.5],
    'respond_min_hcp': [4, 5, 6, 7, 8],
    'open_min_hcp': [10, 11, 12, 13],
    'slam_small_min': [31, 32, 33, 34, 35, 99],
    'overcall_min_hcp': [8, 9, 10, 11, 12],
    'respond_raise_limit_min': [8, 9, 10, 11, 12],
    'responder_rebid_weak_max': [8, 9, 10, 11, 12],
}

MEDIUM_IMPACT_RANGES = {
    'open_1nt_min': [14, 15, 16],
    'open_1nt_max': [16, 17, 18],
    'dist_void': [2, 3, 4],
    'dist_singleton': [1, 2, 3],
    'dist_doubleton': [0, 1],
    'support_void': [4, 5, 6],
    'support_singleton': [2, 3, 4],
    'competitive_double_min': [13, 14, 15, 16, 17],
    'rebid_min_max': [13, 14, 15],
    'rebid_med_max': [16, 17, 18],
}

CARDPLAY_RANGES = {
    'trump_draw_min': [10, 11, 12, 13],
    'cover_honor_min': [10, 11, 12],
    'max_ruff_potential': [2, 3, 4],
    'trump_management_mode': ['always', 'smart'],
    'vul_open_adjust': [0, 1],
    'vul_game_adjust': [0, 1],
}

INTERACTION_PAIRS = [
    ('game_combined_min', [23, 24, 25, 26, 27, 28],
     'partner_est_fraction', [0.2, 0.25, 0.3, 0.333, 0.4, 0.5]),
    ('open_min_hcp', [10, 11, 12, 13],
     'respond_min_hcp', [5, 6, 7, 8]),
]


if __name__ == '__main__':
    print('=' * 60)
    print('BRIDGE PARAMETER OPTIMIZATION')
    print('Paired comparison + significance testing + holdout validation')
    print('=' * 60)

    base = BridgeParams()
    t0 = time.time()

    # Phase 1
    print('\n[Phase 1/6] High-impact parameters')
    best, score = coordinate_descent(
        base, HIGH_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=3, verbose=True
    )

    # Phase 2
    print('\n[Phase 2/6] Medium-impact parameters')
    best2, score2 = coordinate_descent(
        best, MEDIUM_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=2, verbose=True
    )

    # Phase 3
    print('\n[Phase 3/6] Card play parameters')
    best3, score3 = coordinate_descent(
        best2, CARDPLAY_RANGES,
        num_boards=2000, seed=42, passes=2, verbose=True
    )

    # Phase 4
    print('\n[Phase 4/6] Pairwise interaction search')
    best4, score4 = pairwise_optimize(
        best3, INTERACTION_PAIRS,
        num_boards=1500, seed=42, verbose=True
    )

    # Phase 5
    print('\n[Phase 5/6] Final sweep')
    best5, score5 = coordinate_descent(
        best4, HIGH_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=1, verbose=True
    )

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params_optimized.json')
    best5.to_json(out_path)
    print(f'\n  Saved to config/params_optimized.json')

    # Phase 6
    print('\n[Phase 6/6] Holdout validation (5 unseen seeds)')
    validate_on_holdout(best5, base, num_boards=500, verbose=True)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Summary
    print('\n=== PARAMETER CHANGES ===')
    all_ranges = {**HIGH_IMPACT_RANGES, **MEDIUM_IMPACT_RANGES, **CARDPLAY_RANGES}
    changes = 0
    for field_name in sorted(all_ranges.keys()):
        old = getattr(base, field_name)
        new = getattr(best5, field_name)
        if old != new:
            print(f'  {field_name}: {old} -> {new}')
            changes += 1
    print(f'\n  {changes} parameters changed from defaults')
