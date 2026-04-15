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
    # Batch 10: rework knobs.
    mc_budget_seconds: float = 0.5      # wall-clock cap per decision
    mc_use_constraints: bool = True     # reject deals that violate auction
    mc_constraint_max_rejects: int = 50 # fallback: give up after N rejections
    mc_dedupe_equivalents: bool = True  # skip evaluating obviously equivalent cards

    # ── NT-response conventions (Batch 1) ─────────────────────────
    # Measured head-to-head against RuleBasedPlayer: the transfer ladder
    # costs ~10 IMPs/board because post-transfer opener rebids don't yet
    # handle the full 2NT/3NT/3M/4M matrix correctly — natural 1NT-3NT
    # reaches the right strain more often until the rebid ladder is
    # filled out. Default OFF until that work is done; infra preserved
    # for opt-in and future tuning.
    use_stayman: bool = False
    use_jacoby_transfers: bool = False
    # Gerber over 1NT is too risky as an auto-trigger (hard to stop at a
    # sensible level after an unfavorable ace response). 4NT quantitative
    # handles slam tries well enough.
    use_gerber: bool = False
    stayman_min_hcp: int = 8
    transfer_super_accept_min_hcp: int = 17
    transfer_super_accept_min_trumps: int = 4
    gerber_min_hcp: int = 16

    # ── Slam machinery (Batch 2) ──────────────────────────────────
    # Measured -4.5/board in head-to-head. The structure is correct per
    # SAYC but the post-J2NT opener-rebid ladder + splinter-response
    # evaluation need fit/control-count refinement. Infra preserved for
    # opt-in when tuning catches up.
    use_rkcb: bool = False
    use_jacoby_2nt: bool = False
    jacoby_2nt_min_hcp: int = 13
    use_splinters: bool = False
    splinter_min_hcp: int = 13
    splinter_max_hcp: int = 15

    # ── Competitive doubles (Batch 3) ─────────────────────────────
    use_negative_doubles: bool = True
    use_support_doubles: bool = True
    negative_double_min_hcp_1lvl: int = 6
    negative_double_min_hcp_2lvl: int = 8
    negative_double_max_overcall_level: int = 2

    # ── Against-their-opening conventions (Batch 4) ───────────────
    # Takeout X is individually near-neutral, but combined with Michaels
    # and Unusual 2NT the measured effect turns slightly negative
    # (interactions put us in bad competitive contracts). Default off
    # until advancer logic is hardened.
    use_takeout_doubles: bool = False
    use_michaels: bool = False
    use_unusual_2nt: bool = False
    takeout_double_min_hcp: int = 12
    michaels_min_hcp: int = 6
    michaels_max_hcp: int = 11
    unusual_2nt_min_hcp: int = 5
    unusual_2nt_max_hcp: int = 11

    # ── Preempts & weak openings (Batch 5) ────────────────────────
    # Weak 2s and preempts measured net-negative — opponents exploit
    # them consistently and our follow-ups overbid. Default OFF until
    # response structure (feature-ask, quality gate) is refined.
    use_weak_twos: bool = False
    use_preempts: bool = False
    weak_two_min_hcp: int = 6
    weak_two_max_hcp: int = 10
    preempt_3_min_hcp: int = 5
    preempt_3_max_hcp: int = 9
    preempt_4_min_hcp: int = 5
    preempt_4_max_hcp: int = 10
    preempt_raise_game_min_hcp: int = 15

    # ── Minor raises & gadgets (Batch 6) ──────────────────────────
    # Inverted minors showed -2.6/board in multi-seed average, Drury
    # didn't materially change the signal. Default off until the
    # strong-2m continuation (forcing 2NT ask, stopper-showing) is
    # implemented — right now partner passes a strong 2m response.
    use_inverted_minors: bool = False
    use_drury: bool = False
    inverted_strong_min_hcp: int = 11
    inverted_weak_min_hcp: int = 5
    inverted_weak_max_hcp: int = 9
    drury_min_hcp: int = 10
    drury_max_hcp: int = 11

    # ── Defensive signals & opening lead (Batch 7) ────────────────
    # Attitude signals alone measured mildly negative — against
    # RuleBasedPlayer (which doesn't read signals), the information
    # leak to declarer outweighs the benefit to partner. Default off.
    # Count signals leak info in both directions, permanently off by
    # default until a "partner-aware" filter exists.
    use_attitude_signals: bool = False
    use_count_signals: bool = False
    avoid_leading_opp_suit: bool = True    # this one IS helpful
    attitude_encourage_min_rank: int = 12

    # ── Declarer technique (Batch 8) ──────────────────────────────
    # Hold-up in NT measured mildly negative even after tightening;
    # the existing trick-planning logic catches most obvious hold-ups
    # anyway. Default off until we have richer suit-combination
    # information (auction-driven length estimates for the led suit).
    use_hold_up_play: bool = False
    hold_up_max_combined: int = 6
    hold_up_max_rounds: int = 1

    # ── Advanced play (Batch 11) ──────────────────────────────────
    # Unblocking fires rarely (exactly Kx/Ax doubleton under partner's
    # winner) so measured impact is noise-level; left ON since the
    # logic is provably correct in its narrow trigger.
    use_unblocking: bool = True

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
