"""Parameterized configuration for bridge AI agents.

All tunable thresholds live here as a single dataclass. Default values
reproduce the current hardcoded behavior exactly. Parameters can be
loaded from / saved to JSON for empirical optimization.
"""

import json
from dataclasses import dataclass, asdict, fields


@dataclass
class BridgeParams:
    """Every tunable threshold in the bidding and card-play agents."""

    # ── Opening bid thresholds ────────────────────────────────────
    open_min_hcp: int = 12
    open_1nt_min: int = 15
    open_1nt_max: int = 17
    open_2nt_min: int = 20
    open_2nt_max: int = 21
    open_strong_min: int = 22

    # ── Response thresholds ───────────────────────────────────────
    respond_min_hcp: int = 6
    respond_1nt_pass_max: int = 7
    respond_1nt_inv_max: int = 9
    respond_1nt_game_max: int = 15
    respond_2nt_pass_max: int = 3
    respond_2nt_game_max: int = 10
    respond_2c_weak_max: int = 7
    respond_raise_limit_min: int = 10
    respond_raise_game_min: int = 13
    respond_new_2_min: int = 10
    respond_2nt_inv_min: int = 10
    respond_2nt_inv_max: int = 12
    respond_3nt_min: int = 13

    # ── Combined point targets ────────────────────────────────────
    game_combined_min: int = 26
    inv_combined_min: int = 23
    slam_small_min: int = 33
    slam_grand_min: int = 37

    # ── Partner estimation ────────────────────────────────────────
    partner_est_fraction: float = 0.333

    # ── Opener rebid HCP brackets ─────────────────────────────────
    rebid_min_max: int = 14
    rebid_med_max: int = 17
    rebid_strong_max: int = 19

    # ── Overcall thresholds ───────────────────────────────────────
    overcall_min_hcp: int = 10
    overcall_max_hcp: int = 16
    overcall_1nt_min: int = 15
    overcall_1nt_max: int = 17
    overcall_strong_min: int = 17

    # ── Competitive / penalty double ──────────────────────────────
    competitive_double_min: int = 15
    competitive_double_trump_len: int = 4

    # ── Slam investigation ────────────────────────────────────────
    slam_min_qt: float = 2.0
    slam_direct_min_qt: float = 3.0

    # ── Responder rebid ──────────────────────────────────────────
    responder_rebid_weak_max: int = 10
    responder_rebid_fit_min_combined: int = 18

    # ── Hand evaluation point values ──────────────────────────────
    dist_void: int = 3
    dist_singleton: int = 2
    dist_doubleton: int = 1
    support_void: int = 5
    support_singleton: int = 3
    support_doubleton: int = 1

    # ── Card play tactical thresholds ─────────────────────────────
    cover_honor_min: int = 11      # Rank.JACK
    sequence_min_rank: int = 10    # Rank.TEN
    trump_draw_min: int = 12       # Rank.QUEEN

    # ── Monte Carlo card play ─────────────────────────────────────
    use_monte_carlo: bool = False
    monte_carlo_samples: int = 20

    # ── NT-response conventions (Batch 1) ─────────────────────────
    use_stayman: bool = True
    use_jacoby_transfers: bool = True
    use_gerber: bool = True
    stayman_min_hcp: int = 8
    transfer_super_accept_min_hcp: int = 17
    transfer_super_accept_min_trumps: int = 4
    gerber_min_hcp: int = 16

    # ── Slam machinery (Batch 2) ──────────────────────────────────
    use_rkcb: bool = True                # Roman Key Card Blackwood 1430
    use_jacoby_2nt: bool = True          # 1M-2NT game-forcing raise
    jacoby_2nt_min_hcp: int = 13
    use_splinters: bool = True           # double-jump shortness raise
    splinter_min_hcp: int = 13
    splinter_max_hcp: int = 15

    # ── Fit detection ─────────────────────────────────────────────
    fit_min_support_major: int = 3
    fit_min_support_general: int = 4

    # ── Trump management ──────────────────────────────────────────
    trump_management_mode: str = 'always'  # 'always', 'smart', 'never'
    max_ruff_potential: int = 3

    # ── Vulnerability adjustments ─────────────────────────────────
    vul_open_adjust: int = 0    # lower open threshold by N when NV
    vul_game_adjust: int = 0    # lower game threshold by N when NV

    # ── Decision tracing ──────────────────────────────────────────
    trace_enabled: bool = False

    def to_json(self, path: str):
        """Save parameters to a JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'BridgeParams':
        """Load parameters from JSON, using defaults for missing keys."""
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
