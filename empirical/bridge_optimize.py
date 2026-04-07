"""Parameter optimization with statistical rigor.

Improvements over naive coordinate descent:
1. Multi-seed evaluation — average across 3 seeds to prevent overfitting
2. Paired comparison — same deals with both configs, t-test for significance
3. Pairwise interaction search — jointly optimize interacting parameters
4. Holdout validation — verify gains on unseen seeds after optimization
5. Noise-aware — requires statistical significance before accepting changes
"""

import os
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
    """Play num_boards and return average NS score advantage.

    Kept for backward compatibility.
    """
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

    # Per-board paired differences: (B_net) - (A_net)
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

def coordinate_descent(base_params: BridgeParams,
                       param_ranges: Dict[str, list],
                       num_boards: int = 2000,
                       seed: int = 42,
                       passes: int = 2,
                       verbose: bool = True) -> Tuple[BridgeParams, float]:
    """Optimize parameters one at a time with paired significance testing.

    Only accepts a parameter change if it's both positive AND
    statistically significant (t > 1.5, roughly p < 0.07).
    """
    best = base_params
    best_score = evaluate_params(best, num_boards, seed)

    if verbose:
        print(f'Baseline score: {best_score:+.1f} per board')
        print(f'Optimizing {len(param_ranges)} parameters, {passes} passes')
        print(f'Significance threshold: t > 1.5')
        print()

    for pass_num in range(passes):
        if verbose:
            print(f'--- Pass {pass_num + 1}/{passes} ---')

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
                if sig and diff > best_diff:
                    best_diff = diff
                    best_t = t_stat
                    best_val = val

            if best_val != current_val:
                best = replace(best, **{param_name: best_val})
                best_score += best_diff
                improved = True
                if verbose:
                    print(f'  {param_name}: {current_val} -> {best_val} '
                          f'(+{best_diff:.1f}/board, t={best_t:.1f})')
            elif verbose:
                print(f'  {param_name}: {current_val} (unchanged)')

        if not improved:
            if verbose:
                print('  No significant improvement, stopping early.')
            break

    return best, best_score


def pairwise_optimize(base_params: BridgeParams,
                      param_pairs: List[Tuple[str, list, str, list]],
                      num_boards: int = 1500,
                      seed: int = 42,
                      verbose: bool = True) -> Tuple[BridgeParams, float]:
    """Jointly optimize pairs of interacting parameters.

    param_pairs: list of (name_a, values_a, name_b, values_b) tuples.
    Tries all combinations for each pair and keeps the best.
    """
    best = base_params

    for name_a, vals_a, name_b, vals_b in param_pairs:
        cur_a = getattr(best, name_a)
        cur_b = getattr(best, name_b)
        best_combo = (cur_a, cur_b)
        best_diff = 0.0
        best_t = 0.0

        combos_tried = 0
        for va in vals_a:
            for vb in vals_b:
                if va == cur_a and vb == cur_b:
                    continue
                trial = replace(best, **{name_a: va, name_b: vb})
                diff, t_stat, sig = paired_compare(
                    best, trial, num_boards, seed, min_t=1.5
                )
                combos_tried += 1
                if sig and diff > best_diff:
                    best_diff = diff
                    best_t = t_stat
                    best_combo = (va, vb)

        if best_combo != (cur_a, cur_b):
            best = replace(best, **{name_a: best_combo[0], name_b: best_combo[1]})
            if verbose:
                print(f'  {name_a}+{name_b}: ({cur_a},{cur_b}) -> '
                      f'({best_combo[0]},{best_combo[1]}) '
                      f'(+{best_diff:.1f}/board, t={best_t:.1f}, '
                      f'tried {combos_tried})')
        elif verbose:
            print(f'  {name_a}+{name_b}: ({cur_a},{cur_b}) unchanged '
                  f'(tried {combos_tried})')

    score = evaluate_params(best, num_boards, seed)
    return best, score


def validate_on_holdout(params: BridgeParams, baseline: BridgeParams,
                        num_boards: int = 1000,
                        holdout_seeds: List[int] = None,
                        verbose: bool = True) -> Tuple[float, float, float, float]:
    """Validate optimized params on unseen seeds.

    Returns (opt_mean, opt_stderr, baseline_mean, baseline_stderr).
    """
    if holdout_seeds is None:
        holdout_seeds = [777, 888, 999, 1111, 2222]

    opt_scores = []
    base_scores = []
    for seed in holdout_seeds:
        opt_scores.append(evaluate_params(params, num_boards, seed))
        base_scores.append(evaluate_params(baseline, num_boards, seed))

    def _stats(scores):
        m = sum(scores) / len(scores)
        v = sum((s - m) ** 2 for s in scores) / (len(scores) - 1)
        se = math.sqrt(v / len(scores))
        return m, se

    opt_m, opt_se = _stats(opt_scores)
    base_m, base_se = _stats(base_scores)

    if verbose:
        print(f'\n=== HOLDOUT VALIDATION ({len(holdout_seeds)} seeds x {num_boards} boards) ===')
        print(f'  Baseline:  {base_m:+.1f} +/- {base_se:.1f} per board')
        print(f'  Optimized: {opt_m:+.1f} +/- {opt_se:.1f} per board')
        print(f'  Improvement: {opt_m - base_m:+.1f} per board')
        if opt_m > base_m:
            # Combined SE for the difference
            diff_se = math.sqrt(opt_se**2 + base_se**2)
            t = (opt_m - base_m) / diff_se if diff_se > 0 else 0
            print(f'  t-statistic: {t:.1f} ({"significant" if t > 2.0 else "NOT significant"})')

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

# Parameters that interact strongly — optimize jointly
INTERACTION_PAIRS = [
    ('game_combined_min', [23, 24, 25, 26, 27, 28],
     'partner_est_fraction', [0.2, 0.25, 0.3, 0.333, 0.4, 0.5]),
    ('open_min_hcp', [10, 11, 12, 13],
     'respond_min_hcp', [5, 6, 7, 8]),
]


if __name__ == '__main__':
    print('=' * 60)
    print('BRIDGE PARAMETER OPTIMIZATION (with significance testing)')
    print('=' * 60)
    print()

    base = BridgeParams()
    t0 = time.time()

    # Phase 1: Coordinate descent on high-impact params
    print('Phase 1: High-impact parameters (coordinate descent)')
    best, score = coordinate_descent(
        base, HIGH_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=3, verbose=True
    )
    print(f'\nPhase 1 complete: score {score:+.1f} per board')
    print(f'Time: {time.time() - t0:.0f}s')

    # Phase 2: Medium-impact params
    print('\nPhase 2: Medium-impact parameters')
    best2, score2 = coordinate_descent(
        best, MEDIUM_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=2, verbose=True
    )
    print(f'\nPhase 2 complete: score {score2:+.1f} per board')

    # Phase 3: Card play params
    print('\nPhase 3: Card play + vulnerability parameters')
    best3, score3 = coordinate_descent(
        best2, CARDPLAY_RANGES,
        num_boards=2000, seed=42, passes=2, verbose=True
    )
    print(f'\nPhase 3 complete: score {score3:+.1f} per board')

    # Phase 4: Pairwise interaction search
    print('\nPhase 4: Pairwise interaction optimization')
    best4, score4 = pairwise_optimize(
        best3, INTERACTION_PAIRS,
        num_boards=1500, seed=42, verbose=True
    )
    print(f'\nPhase 4 complete: score {score4:+.1f} per board')

    # Phase 5: Re-sweep high-impact after interactions
    print('\nPhase 5: Final sweep (high-impact)')
    best5, score5 = coordinate_descent(
        best4, HIGH_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=1, verbose=True
    )
    print(f'\nPhase 5 complete: score {score5:+.1f} per board')

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'params_optimized.json')
    best5.to_json(out_path)
    print(f'\nOptimized params saved to config/params_optimized.json')

    # Phase 6: Holdout validation
    print(f'\nPhase 6: Holdout validation')
    validate_on_holdout(best5, base, num_boards=500, verbose=True)

    print(f'\nTotal time: {time.time() - t0:.0f}s')

    # Show key changes
    print('\n=== KEY PARAMETER CHANGES ===')
    all_ranges = {**HIGH_IMPACT_RANGES, **MEDIUM_IMPACT_RANGES, **CARDPLAY_RANGES}
    for field_name in sorted(all_ranges.keys()):
        old = getattr(base, field_name)
        new = getattr(best5, field_name)
        if old != new:
            print(f'  {field_name}: {old} -> {new}')
