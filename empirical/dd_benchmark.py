"""Benchmark AI play quality against double-dummy optimal.

Runs boards with AI players, then solves each board DD to measure
the trick gap (how many tricks the AI leaves on the table).
"""

import random
import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from engine.card import Card, Suit, deal
from engine.state import GameState
from engine.game import VUL_SCHEDULE
from engine.dds import double_dummy_tricks


@dataclass
class BoardResult:
    """Result of one board: AI vs DD comparison."""
    board_num: int = 0
    ai_tricks: int = 0
    dd_tricks: int = 0
    trick_gap: int = 0       # dd_tricks - ai_tricks (>= 0 means AI lost tricks)
    contract_level: int = 0
    contract_strain: Optional[int] = None
    declarer: int = -1
    passed_out: bool = False


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""
    boards: List[BoardResult] = field(default_factory=list)
    total_boards: int = 0
    played_boards: int = 0
    mean_trick_gap: float = 0.0
    median_trick_gap: float = 0.0
    pct_optimal: float = 0.0     # % of boards where gap == 0
    mean_ai_tricks: float = 0.0
    mean_dd_tricks: float = 0.0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== DD Benchmark: {self.total_boards} boards ===",
            f"  Played (non-passed): {self.played_boards}",
            f"  Mean AI tricks:   {self.mean_ai_tricks:.2f}",
            f"  Mean DD tricks:   {self.mean_dd_tricks:.2f}",
            f"  Mean trick gap:   {self.mean_trick_gap:.2f}",
            f"  Median trick gap: {self.median_trick_gap:.1f}",
            f"  Optimal play %:   {self.pct_optimal:.1f}%",
            f"  Time: {self.elapsed_seconds:.1f}s",
        ]
        return '\n'.join(lines)

    def worst_boards(self, n: int = 10) -> List[BoardResult]:
        """Return the N boards with the largest trick gap."""
        played = [b for b in self.boards if not b.passed_out]
        return sorted(played, key=lambda b: b.trick_gap, reverse=True)[:n]


def benchmark_vs_dd(player_factory: Callable,
                    num_boards: int = 100,
                    seed: int = 42) -> BenchmarkResult:
    """Run boards with AI, compare to DD optimal.

    Args:
        player_factory: Callable returning list of 4 Player instances.
        num_boards: Number of boards to play.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkResult with per-board and aggregate stats.
    """
    random.seed(seed)
    t0 = time.time()
    boards = []

    for board_idx in range(num_boards):
        board_num = board_idx + 1
        vul = VUL_SCHEDULE[(board_num - 1) % 16]
        dealer = (board_num - 1) % 4

        gs = GameState(board_num=board_num, vulnerable=vul, dealer=dealer)
        players = {i: p for i, p in enumerate(player_factory())}

        with contextlib.redirect_stdout(io.StringIO()):
            gs.new_deal()

        # Save hands before play
        saved_hands = {s: list(gs.hands[s]) for s in range(4)}

        # Play the board with AI
        with contextlib.redirect_stdout(io.StringIO()):
            while gs.phase == 'AUCTION':
                a = gs.next_actor()
                obs = gs.observation(a)
                gs.apply_call(players[a].bid(obs))

            while gs.phase == 'PLAY':
                a = gs.next_actor()
                obs = gs.observation(a)
                gs.play_card(a, players[a].play_card(obs))

        contract = gs.auction.contract
        br = BoardResult(board_num=board_num)

        if contract is None:
            br.passed_out = True
        else:
            br.contract_level = contract.level
            br.contract_strain = int(contract.strain)
            br.declarer = gs.auction.declarer

            # AI tricks
            if gs.play:
                pr = gs.play.result()
                br.ai_tricks = pr['declarer_tricks']

            # DD optimal tricks for same contract
            trump = contract.strain if contract.strain != Suit.NT else None
            br.dd_tricks = double_dummy_tricks(
                saved_hands, trump, br.declarer
            )
            br.trick_gap = br.dd_tricks - br.ai_tricks

        boards.append(br)

    # Aggregate
    played = [b for b in boards if not b.passed_out]
    result = BenchmarkResult(
        boards=boards,
        total_boards=num_boards,
        played_boards=len(played),
        elapsed_seconds=time.time() - t0,
    )

    if played:
        gaps = [b.trick_gap for b in played]
        result.mean_trick_gap = sum(gaps) / len(gaps)
        sorted_gaps = sorted(gaps)
        mid = len(sorted_gaps) // 2
        result.median_trick_gap = sorted_gaps[mid]
        result.pct_optimal = sum(1 for g in gaps if g == 0) / len(gaps) * 100
        result.mean_ai_tricks = sum(b.ai_tricks for b in played) / len(played)
        result.mean_dd_tricks = sum(b.dd_tricks for b in played) / len(played)

    return result


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ai.smart_player import SmartPlayer
    from engine.player import RuleBasedPlayer

    print("Running DD benchmark: SmartPlayer vs RuleBasedPlayer (100 boards)...")
    result = benchmark_vs_dd(
        player_factory=lambda: [
            SmartPlayer(0), RuleBasedPlayer(1),
            SmartPlayer(2), RuleBasedPlayer(3),
        ],
        num_boards=100,
        seed=42,
    )
    print(result.summary())
    print("\nWorst 5 boards:")
    for b in result.worst_boards(5):
        print(f"  Board {b.board_num}: AI={b.ai_tricks} DD={b.dd_tricks} "
              f"gap={b.trick_gap} level={b.contract_level}")
