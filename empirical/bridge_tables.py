"""Empirical lookup tables built from mass simulation data.

Provides EV-based game/slam decisions and Bayesian partner HCP
distributions derived from actual game outcomes.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from .bridge_stats import BoardRecord


def build_game_ev_table(records: List[BoardRecord]) -> Dict[Tuple, float]:
    """Build a table of average score by (combined_hcp_bin, vulnerability).

    Returns dict mapping (hcp_bin, vul) → average score when game was bid.
    Positive = profitable to bid game, negative = better to stay low.
    """
    # Separate game bids vs part-score bids
    game_scores = defaultdict(list)
    part_scores = defaultdict(list)

    for r in records:
        if r.passed_out or r.declarer < 0:
            continue

        dec_ns = r.declarer in (0, 2)
        combined_hcp = r.ns_combined_hcp if dec_ns else r.ew_combined_hcp
        vul = r.vul_ns if dec_ns else r.vul_ew
        score = r.score_ns if dec_ns else r.score_ew
        hcp_bin = combined_hcp // 2 * 2

        if r.contract_level >= 3:
            game_scores[(hcp_bin, vul)].append(score)
        else:
            part_scores[(hcp_bin, vul)].append(score)

    # Compute EV difference
    ev_table = {}
    all_keys = set(game_scores.keys()) | set(part_scores.keys())
    for key in all_keys:
        gs = game_scores.get(key, [])
        ps = part_scores.get(key, [])
        avg_game = sum(gs) / len(gs) if gs else 0
        avg_part = sum(ps) / len(ps) if ps else 0
        ev_table[key] = avg_game - avg_part

    return ev_table


def build_slam_ev_table(records: List[BoardRecord]) -> Dict[Tuple, float]:
    """Build a table of slam EV by (combined_hcp_bin, has_fit, vulnerability)."""
    slam_scores = defaultdict(list)
    game_scores = defaultdict(list)

    for r in records:
        if r.passed_out or r.declarer < 0:
            continue

        dec_ns = r.declarer in (0, 2)
        combined_hcp = r.ns_combined_hcp if dec_ns else r.ew_combined_hcp
        fit_len = r.ns_best_fit_len if dec_ns else r.ew_best_fit_len
        has_fit = fit_len >= 8
        vul = r.vul_ns if dec_ns else r.vul_ew
        score = r.score_ns if dec_ns else r.score_ew
        hcp_bin = combined_hcp // 2 * 2

        if r.contract_level >= 6:
            slam_scores[(hcp_bin, has_fit, vul)].append(score)
        elif r.contract_level >= 3:
            game_scores[(hcp_bin, has_fit, vul)].append(score)

    ev_table = {}
    all_keys = set(slam_scores.keys()) | set(game_scores.keys())
    for key in all_keys:
        ss = slam_scores.get(key, [])
        gs = game_scores.get(key, [])
        avg_slam = sum(ss) / len(ss) if ss else 0
        avg_game = sum(gs) / len(gs) if gs else 0
        ev_table[key] = avg_slam - avg_game

    return ev_table


def build_make_rate_table(records: List[BoardRecord]) -> Dict[Tuple, Tuple[float, int]]:
    """Build table of (make_rate, sample_count) by (combined_hcp_bin, contract_level, vul).

    Returns dict mapping (hcp_bin, level, vul) → (make_rate, count).
    """
    buckets = defaultdict(lambda: {'made': 0, 'total': 0})

    for r in records:
        if r.passed_out or r.declarer < 0:
            continue

        dec_ns = r.declarer in (0, 2)
        combined_hcp = r.ns_combined_hcp if dec_ns else r.ew_combined_hcp
        vul = r.vul_ns if dec_ns else r.vul_ew
        hcp_bin = combined_hcp // 2 * 2

        key = (hcp_bin, r.contract_level, vul)
        buckets[key]['total'] += 1
        buckets[key]['made'] += int(r.made_contract)

    return {k: (v['made'] / v['total'], v['total'])
            for k, v in buckets.items() if v['total'] >= 3}


def build_partner_hcp_distributions(records: List[BoardRecord]) -> Dict[str, List[float]]:
    """Build empirical P(partner_hcp | bid_category) distributions.

    Analyzes what HCP partners actually held when they made various
    types of bids, producing 41-element probability vectors.

    Returns dict mapping category string → list of 41 floats (probabilities for HCP 0-40).
    """
    # We approximate by looking at what HCP the declaring side's
    # non-declarer had (partner of opener or responder)
    histograms = defaultdict(lambda: [0] * 41)

    for r in records:
        if r.passed_out or r.declarer < 0:
            continue

        dec_ns = r.declarer in (0, 2)

        if dec_ns:
            declarer_hcp = r.hcp_n if r.declarer == 0 else r.hcp_s
            partner_hcp = r.hcp_s if r.declarer == 0 else r.hcp_n
        else:
            declarer_hcp = r.hcp_e if r.declarer == 1 else r.hcp_w
            partner_hcp = r.hcp_w if r.declarer == 1 else r.hcp_e

        # Categorize by contract level (rough proxy for bid type)
        if r.contract_level == 1:
            cat = 'respond_low'
        elif r.contract_level == 2:
            cat = 'respond_mid'
        elif r.contract_level >= 3 and r.contract_strain == 4:  # NT
            cat = 'respond_game_nt'
        elif r.contract_level >= 4:
            cat = 'respond_game_suit'
        else:
            cat = 'respond_other'

        if 0 <= partner_hcp <= 40:
            histograms[cat][partner_hcp] += 1

    # Normalize to probabilities
    distributions = {}
    for cat, hist in histograms.items():
        total = sum(hist)
        if total > 0:
            distributions[cat] = [h / total for h in hist]

    return distributions


def expected_hcp_from_dist(dist: List[float]) -> float:
    """Compute E[HCP] from a probability distribution."""
    return sum(h * p for h, p in enumerate(dist))


def variance_from_dist(dist: List[float]) -> float:
    """Compute Var[HCP] from a probability distribution."""
    mean = expected_hcp_from_dist(dist)
    return sum((h - mean) ** 2 * p for h, p in enumerate(dist))


class EmpiricalTables:
    """Container for all empirical lookup tables."""

    def __init__(self, records: List[BoardRecord]):
        self.game_ev = build_game_ev_table(records)
        self.slam_ev = build_slam_ev_table(records)
        self.make_rates = build_make_rate_table(records)
        self.partner_dists = build_partner_hcp_distributions(records)

    def game_is_profitable(self, combined_hcp: int, vul: bool) -> bool:
        """Should we bid game with this combined HCP and vulnerability?"""
        hcp_bin = combined_hcp // 2 * 2
        ev = self.game_ev.get((hcp_bin, vul))
        if ev is None:
            return combined_hcp >= 26  # fallback
        return ev > 0

    def slam_is_profitable(self, combined_hcp: int, has_fit: bool,
                           vul: bool) -> bool:
        """Should we bid slam?"""
        hcp_bin = combined_hcp // 2 * 2
        ev = self.slam_ev.get((hcp_bin, has_fit, vul))
        if ev is None:
            return combined_hcp >= 33  # fallback
        return ev > 0

    def make_rate(self, combined_hcp: int, level: int, vul: bool) -> float:
        """Empirical probability of making a contract at this level."""
        hcp_bin = combined_hcp // 2 * 2
        result = self.make_rates.get((hcp_bin, level, vul))
        if result is None:
            return 0.5  # no data
        return result[0]

    def partner_expected_hcp(self, category: str) -> float:
        """Expected partner HCP for a given bid category."""
        dist = self.partner_dists.get(category)
        if dist is None:
            return 10.0  # neutral fallback
        return expected_hcp_from_dist(dist)
