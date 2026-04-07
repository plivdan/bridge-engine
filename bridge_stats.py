"""Mass simulation data collection for empirical parameter tuning.

Runs thousands of boards and records per-hand feature vectors including
HCP, shape, distribution, fit length, contract outcome, and score.
"""

import random
import csv
import contextlib
import io
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable

from card import Card, Suit, Rank, deal
from auction import AuctionState
from hand_eval import (
    hcp, hand_shape, total_points, distribution_points,
    losing_trick_count, quick_tricks, suit_length,
)
from state import GameState
from game import VUL_SCHEDULE


@dataclass
class BoardRecord:
    """Feature vector for a single completed board."""
    board_num: int = 0
    dealer: int = 0
    vul_ns: bool = False
    vul_ew: bool = False

    # Per-seat metrics (indexed 0=N, 1=E, 2=S, 3=W)
    hcp_n: int = 0; hcp_e: int = 0; hcp_s: int = 0; hcp_w: int = 0
    tp_n: int = 0; tp_e: int = 0; tp_s: int = 0; tp_w: int = 0
    ltc_n: int = 0; ltc_e: int = 0; ltc_s: int = 0; ltc_w: int = 0
    qt_n: float = 0; qt_e: float = 0; qt_s: float = 0; qt_w: float = 0

    # Partnership features
    ns_combined_hcp: int = 0
    ew_combined_hcp: int = 0
    ns_combined_tp: int = 0
    ew_combined_tp: int = 0
    ns_best_fit_suit: int = -1  # Suit value or -1
    ns_best_fit_len: int = 0
    ew_best_fit_suit: int = -1
    ew_best_fit_len: int = 0

    # Auction outcome
    contract_level: int = 0  # 0 = passed out
    contract_strain: int = -1  # Suit value
    declarer: int = -1
    doubled: int = 0
    passed_out: bool = False
    auction_length: int = 0

    # Play outcome
    tricks_declarer: int = 0
    tricks_ns: int = 0
    tricks_ew: int = 0
    made_contract: bool = False
    overtricks: int = 0

    # Score
    score_ns: int = 0
    score_ew: int = 0


def _best_fit(hand_a: List[Card], hand_b: List[Card]):
    """Return (best_suit, combined_length) for two partnership hands."""
    best_suit = -1
    best_len = 0
    for suit in (Suit.S, Suit.H, Suit.D, Suit.C):
        combined = suit_length(hand_a, suit) + suit_length(hand_b, suit)
        if combined > best_len:
            best_len = combined
            best_suit = int(suit)
    return best_suit, best_len


def collect_boards(player_factory: Callable, num_boards: int,
                   seed: int = 0) -> List[BoardRecord]:
    """Run *num_boards* and return structured records.

    Args:
        player_factory: callable returning a list of 4 Player instances.
        num_boards: how many boards to simulate.
        seed: random seed for reproducibility.

    Returns:
        List of BoardRecord, one per board.
    """
    random.seed(seed)
    records = []

    for board_idx in range(num_boards):
        board_num = board_idx + 1
        vul = VUL_SCHEDULE[(board_num - 1) % 16]
        dealer = (board_num - 1) % 4

        gs = GameState(board_num=board_num, vulnerable=vul, dealer=dealer)
        players = {i: p for i, p in enumerate(player_factory())}

        with contextlib.redirect_stdout(io.StringIO()):
            gs.new_deal()

        # Compute hand features for all seats
        h = {i: gs.hands[i] for i in range(4)}
        hcp_vals = {i: hcp(h[i]) for i in range(4)}
        tp_vals = {i: total_points(h[i]) for i in range(4)}
        ltc_vals = {i: losing_trick_count(h[i]) for i in range(4)}
        qt_vals = {i: quick_tricks(h[i]) for i in range(4)}

        ns_fit_suit, ns_fit_len = _best_fit(h[0], h[2])
        ew_fit_suit, ew_fit_len = _best_fit(h[1], h[3])

        # Play the board
        with contextlib.redirect_stdout(io.StringIO()):
            while gs.phase == 'AUCTION':
                a = gs.next_actor()
                obs = gs.observation(a)
                gs.apply_call(players[a].bid(obs))

            while gs.phase == 'PLAY':
                a = gs.next_actor()
                obs = gs.observation(a)
                gs.play_card(a, players[a].play_card(obs))

        # Extract results
        contract = gs.auction.contract
        rec = BoardRecord(
            board_num=board_num,
            dealer=dealer,
            vul_ns=vul['NS'],
            vul_ew=vul['EW'],
            hcp_n=hcp_vals[0], hcp_e=hcp_vals[1],
            hcp_s=hcp_vals[2], hcp_w=hcp_vals[3],
            tp_n=tp_vals[0], tp_e=tp_vals[1],
            tp_s=tp_vals[2], tp_w=tp_vals[3],
            ltc_n=ltc_vals[0], ltc_e=ltc_vals[1],
            ltc_s=ltc_vals[2], ltc_w=ltc_vals[3],
            qt_n=qt_vals[0], qt_e=qt_vals[1],
            qt_s=qt_vals[2], qt_w=qt_vals[3],
            ns_combined_hcp=hcp_vals[0] + hcp_vals[2],
            ew_combined_hcp=hcp_vals[1] + hcp_vals[3],
            ns_combined_tp=tp_vals[0] + tp_vals[2],
            ew_combined_tp=tp_vals[1] + tp_vals[3],
            ns_best_fit_suit=ns_fit_suit,
            ns_best_fit_len=ns_fit_len,
            ew_best_fit_suit=ew_fit_suit,
            ew_best_fit_len=ew_fit_len,
            auction_length=len(gs.auction.calls),
            score_ns=gs.score_ns,
            score_ew=gs.score_ew,
        )

        if contract is None:
            rec.passed_out = True
        else:
            rec.contract_level = contract.level
            rec.contract_strain = int(contract.strain)
            rec.declarer = gs.auction.declarer
            rec.doubled = gs.auction.result()['doubled']
            if gs.play:
                pr = gs.play.result()
                rec.tricks_declarer = pr['declarer_tricks']
                rec.tricks_ns = pr['tricks_ns']
                rec.tricks_ew = pr['tricks_ew']
                target = contract.level + 6
                rec.made_contract = rec.tricks_declarer >= target
                rec.overtricks = rec.tricks_declarer - target

        records.append(rec)

    return records


def records_to_csv(records: List[BoardRecord], path: str):
    """Write records to CSV."""
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def records_from_csv(path: str) -> List[BoardRecord]:
    """Read records from CSV."""
    records = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            for k in row:
                if k.startswith('vul_') or k in ('passed_out', 'made_contract'):
                    row[k] = row[k] == 'True'
                elif k.startswith('qt_'):
                    row[k] = float(row[k])
                else:
                    try:
                        row[k] = int(row[k])
                    except (ValueError, TypeError):
                        pass
            records.append(BoardRecord(**row))
    return records


def analyze_records(records: List[BoardRecord]) -> dict:
    """Compute summary statistics from collected records."""
    n = len(records)
    if n == 0:
        return {}

    passed = sum(1 for r in records if r.passed_out)
    played = [r for r in records if not r.passed_out]
    made = sum(1 for r in played if r.made_contract)
    games = [r for r in played if r.contract_level >= 3]
    games_made = sum(1 for r in games if r.made_contract)
    slams = [r for r in played if r.contract_level >= 6]
    slams_made = sum(1 for r in slams if r.made_contract)

    # Tricks vs HCP correlation for declarer
    ns_declaring = [r for r in played if r.declarer in (0, 2)]
    ew_declaring = [r for r in played if r.declarer in (1, 3)]

    # Game success by combined HCP (for NS declaring)
    ns_games = [r for r in ns_declaring if r.contract_level >= 3]
    game_by_hcp = {}
    for r in ns_games:
        bucket = r.ns_combined_hcp // 2 * 2
        game_by_hcp.setdefault(bucket, {'total': 0, 'made': 0, 'score_sum': 0})
        game_by_hcp[bucket]['total'] += 1
        game_by_hcp[bucket]['made'] += int(r.made_contract)
        game_by_hcp[bucket]['score_sum'] += r.score_ns

    return {
        'total_boards': n,
        'passed_out': passed,
        'played': len(played),
        'made_pct': made / len(played) * 100 if played else 0,
        'games': len(games),
        'games_made_pct': games_made / len(games) * 100 if games else 0,
        'slams': len(slams),
        'slams_made_pct': slams_made / len(slams) * 100 if slams else 0,
        'game_by_hcp': game_by_hcp,
    }
