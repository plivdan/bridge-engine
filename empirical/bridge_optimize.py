"""Parameter optimization via coordinate descent.

Evaluates BridgeParams configurations by playing head-to-head
games and selecting the parameters that maximize score.
"""

import os
import random
import contextlib
import io
import time
from dataclasses import replace
from typing import Dict, List, Tuple

from ai.bridge_params import BridgeParams
from ai.smart_player import SmartPlayer
from engine.player import RuleBasedPlayer
from engine.game import Game


def evaluate_params(params: BridgeParams, num_boards: int = 2000,
                    seed: int = 42, vs_default: bool = True) -> float:
    """Play num_boards and return average NS score advantage.

    If vs_default: SmartPlayer(params) NS vs RuleBasedPlayer EW.
    Otherwise: SmartPlayer(params) all 4 seats (self-play baseline).
    """
    random.seed(seed)
    if vs_default:
        players = [
            SmartPlayer(0, params=params),
            RuleBasedPlayer(1),
            SmartPlayer(2, params=params),
            RuleBasedPlayer(3),
        ]
    else:
        players = [SmartPlayer(i, params=params) for i in range(4)]

    with contextlib.redirect_stdout(io.StringIO()):
        results = Game(players, num_boards=num_boards).run()

    ns = sum(r['score_ns'] for r in results)
    ew = sum(r['score_ew'] for r in results)
    return (ns - ew) / num_boards  # per-board advantage


def coordinate_descent(base_params: BridgeParams,
                       param_ranges: Dict[str, list],
                       num_boards: int = 2000,
                       seed: int = 42,
                       passes: int = 2,
                       verbose: bool = True) -> Tuple[BridgeParams, float]:
    """Optimize parameters one at a time, keeping best value.

    Args:
        base_params: Starting parameter configuration.
        param_ranges: Dict of param_name → list of values to try.
        num_boards: Boards per evaluation.
        seed: Random seed for reproducibility.
        passes: Number of full sweeps over all parameters.
        verbose: Print progress.

    Returns:
        (best_params, best_score) tuple.
    """
    best = base_params
    best_score = evaluate_params(best, num_boards, seed)

    if verbose:
        print(f'Baseline score: {best_score:+.1f} per board')
        print(f'Optimizing {len(param_ranges)} parameters, {passes} passes')
        print()

    for pass_num in range(passes):
        if verbose:
            print(f'--- Pass {pass_num + 1}/{passes} ---')

        improved = False
        for param_name, values in param_ranges.items():
            current_val = getattr(best, param_name)
            best_val = current_val
            best_param_score = best_score

            for val in values:
                if val == current_val:
                    continue
                trial = replace(best, **{param_name: val})
                score = evaluate_params(trial, num_boards, seed)
                if score > best_param_score:
                    best_param_score = score
                    best_val = val

            if best_val != current_val:
                best = replace(best, **{param_name: best_val})
                best_score = best_param_score
                improved = True
                if verbose:
                    print(f'  {param_name}: {current_val} → {best_val} '
                          f'(score {best_score:+.1f})')
            elif verbose:
                print(f'  {param_name}: {current_val} (unchanged)')

        if not improved:
            if verbose:
                print('  No improvement this pass, stopping early.')
            break

    return best, best_score


# Default parameter ranges for the highest-impact parameters
HIGH_IMPACT_RANGES = {
    'game_combined_min': [23, 24, 25, 26, 27, 28, 29, 30],
    'partner_est_fraction': [0.2, 0.25, 0.3, 0.333, 0.4, 0.5],
    'respond_min_hcp': [4, 5, 6, 7, 8],
    'open_min_hcp': [10, 11, 12, 13],
    'slam_small_min': [31, 32, 33, 34, 35, 99],  # 99 = effectively disable slams
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


if __name__ == '__main__':
    print('=== BRIDGE PARAMETER OPTIMIZATION ===')
    print()

    base = BridgeParams()
    t0 = time.time()

    # Phase 1: High-impact parameters
    print('Phase 1: High-impact parameters')
    best, score = coordinate_descent(
        base, HIGH_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=3, verbose=True
    )
    print(f'\nPhase 1 complete: score {score:+.1f} per board')
    print(f'Time: {time.time() - t0:.0f}s')

    # Phase 2: Medium-impact parameters
    print('\nPhase 2: Medium-impact parameters')
    best2, score2 = coordinate_descent(
        best, MEDIUM_IMPACT_RANGES,
        num_boards=2000, seed=42, passes=2, verbose=True
    )
    print(f'\nPhase 2 complete: score {score2:+.1f} per board')

    # Save optimized params
    best2.to_json(os.path.join(os.path.dirname(__file__), '..', 'config', 'params_optimized.json'))
    print(f'\nOptimized params saved to config/params_optimized.json')
    print(f'Total time: {time.time() - t0:.0f}s')

    # Show key changes
    print('\n=== KEY PARAMETER CHANGES ===')
    for field_name in sorted(HIGH_IMPACT_RANGES.keys()):
        old = getattr(base, field_name)
        new = getattr(best2, field_name)
        if old != new:
            print(f'  {field_name}: {old} → {new}')
    for field_name in sorted(MEDIUM_IMPACT_RANGES.keys()):
        old = getattr(base, field_name)
        new = getattr(best2, field_name)
        if old != new:
            print(f'  {field_name}: {old} → {new}')
