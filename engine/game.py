"""Bridge game orchestration: batch duplicate-bridge sessions and RL training environment.

This module provides two top-level interfaces for running bridge games:

- :class:`Game`: plays a fixed number of boards with four :class:`Player` instances
  and accumulates scored results, suitable for simulation or evaluation runs.
- :class:`SelfPlayEnv`: a gym-style environment that drives a single board at a time
  via ``reset()`` / ``step()`` calls, suitable for reinforcement-learning training loops.

Module-level constant
---------------------
VUL_SCHEDULE:
    The canonical 16-board vulnerability rotation used in duplicate bridge.
    Entry ``i`` (0-indexed) gives the vulnerability for board number ``i + 1``.
    The schedule repeats with period 16, so board *n* uses entry
    ``(n - 1) % 16``.
"""

from typing import List, Dict, Type
from .state import GameState
from .player import Player
from .auction import PASS

SEATS = ['N', 'E', 'S', 'W']
VUL_SCHEDULE = [
    {'NS': False, 'EW': False},
    {'NS': True,  'EW': False},
    {'NS': False, 'EW': True},
    {'NS': True,  'EW': True},
    {'NS': True,  'EW': False},
    {'NS': False, 'EW': True},
    {'NS': True,  'EW': True},
    {'NS': False, 'EW': False},
    {'NS': False, 'EW': True},
    {'NS': True,  'EW': True},
    {'NS': False, 'EW': False},
    {'NS': True,  'EW': False},
    {'NS': True,  'EW': True},
    {'NS': False, 'EW': False},
    {'NS': True,  'EW': False},
    {'NS': False, 'EW': True},
]

class Game:
    """Runs a duplicate-bridge session consisting of one or more boards.

    Four players are seated North (0), East (1), South (2), West (3).
    Each board uses the standard 16-board vulnerability schedule and rotates
    the dealer seat.  After all boards are played, cumulative NS and EW scores
    are printed to stdout and the per-board results list is returned.

    Attributes:
        players: Mapping from seat index (0-3) to the corresponding
            :class:`Player` instance.
        num_boards: Total number of boards to play in the session.
        results: List of result dicts populated by :meth:`run`.  Each dict
            contains ``board_num``, ``score_ns``, ``score_ew``, ``contract``,
            ``declarer``, and ``doubled``.
    """

    def __init__(self, players: List[Player], num_boards: int = 1):
        """Initialises the session.

        Args:
            players: Exactly four :class:`Player` instances ordered by seat
                (North, East, South, West).
            num_boards: Number of boards to play.  Defaults to ``1``.

        Raises:
            AssertionError: If ``players`` does not contain exactly four entries.
        """
        assert len(players) == 4
        self.players = {i: players[i] for i in range(4)}
        self.num_boards = num_boards
        self.results = []

    def run(self):
        """Plays all boards in the session and returns the results.

        Iterates over ``self.num_boards`` boards, invoking :meth:`_play_board`
        for each one.  After the final board, prints a summary line with
        cumulative NS and EW totals to stdout.

        Returns:
            A list of result dicts, one per board.  Each dict has the keys:

            - ``board_num`` (int): 1-based board number.
            - ``score_ns`` (int): North-South score for the board.
            - ``score_ew`` (int): East-West score for the board.
            - ``contract`` (str | None): The final contract string, or
              ``None`` if all four players passed.
            - ``declarer`` (int): Seat index (0-3) of the declarer.
            - ``doubled`` (int): Doubling level — ``0`` undoubled, ``1``
              doubled, ``2`` redoubled.
        """
        for board in range(self.num_boards):
            result = self._play_board(board + 1)
            self.results.append(result)
        print(f"\n{'='*60}")
        print(f"[SESSION RESULTS] {self.num_boards} boards")
        total_ns = sum(r['score_ns'] for r in self.results)
        total_ew = sum(r['score_ew'] for r in self.results)
        print(f"  Total NS: {total_ns}  Total EW: {total_ew}")
        return self.results

    def _play_board(self, board_num: int):
        """Plays a single board end-to-end and returns its scored result.

        Constructs a fresh :class:`GameState` with the correct vulnerability
        and dealer derived from ``board_num``, deals the cards, then drives
        the auction and play phases to completion by polling each player in
        turn.

        Args:
            board_num: 1-based board number used to look up vulnerability
                (``VUL_SCHEDULE[(board_num - 1) % 16]``) and dealer seat
                (``(board_num - 1) % 4``).

        Returns:
            A dict with keys ``board_num``, ``score_ns``, ``score_ew``,
            ``contract``, ``declarer``, and ``doubled`` (see :meth:`run`
            for full descriptions).
        """
        vul = VUL_SCHEDULE[(board_num - 1) % 16]
        dealer = (board_num - 1) % 4
        gs = GameState(board_num=board_num, vulnerable=vul, dealer=dealer)
        gs.new_deal()

        while gs.phase == 'AUCTION':
            actor = gs.next_actor()
            obs = gs.observation(actor)
            call = self.players[actor].bid(obs)
            gs.apply_call(call)

        while gs.phase == 'PLAY':
            actor = gs.next_actor()
            obs = gs.observation(actor)
            card = self.players[actor].play_card(obs)
            gs.play_card(actor, card)

        return {
            'board_num': board_num,
            'score_ns': gs.score_ns,
            'score_ew': gs.score_ew,
            'contract': gs.auction.contract,
            'declarer': gs.auction.declarer,
            'doubled': gs.auction.result()['doubled'] if gs.auction.contract else 0,
        }


class SelfPlayEnv:
    """Gym-style environment for reinforcement-learning training on bridge.

    The environment manages a single board at a time.  Each call to
    :meth:`reset` starts a new board (incrementing ``board_num`` so that the
    vulnerability schedule advances correctly).  :meth:`step` asks the
    currently active player for its action, applies it to the game state, and
    returns the standard ``(observation, reward, done)`` triple.

    Reward convention:
        - ``(0, 0)`` on every non-terminal step.
        - ``(score_ns, score_ew)`` on the terminal step where ``done=True``.

    Attributes:
        players: Mapping from seat index (0-3) to the corresponding
            :class:`Player` instance.
        board_num: Running count of boards played; incremented by
            :meth:`reset`.  Used to look up the vulnerability schedule.
        gs: The active :class:`GameState`, or ``None`` before the first
            :meth:`reset` call.
    """

    def __init__(self, players: List[Player]):
        """Initialises the environment with four players.

        Args:
            players: Exactly four :class:`Player` instances ordered by seat
                (North, East, South, West).

        Raises:
            AssertionError: If ``players`` does not contain exactly four entries.
        """
        assert len(players) == 4
        self.players = {i: players[i] for i in range(4)}
        self.board_num = 0
        self.gs = None

    def reset(self):
        """Starts a new board and returns the first observation.

        Increments ``board_num``, looks up the corresponding vulnerability
        from :data:`VUL_SCHEDULE`, constructs a fresh :class:`GameState`,
        deals the cards, and returns the observation for the first player to
        act (the dealer in the auction).

        Returns:
            The observation dict produced by ``GameState.observation()`` for
            the first active player.
        """
        self.board_num += 1
        vul = VUL_SCHEDULE[(self.board_num - 1) % 16]
        dealer = (self.board_num - 1) % 4
        self.gs = GameState(board_num=self.board_num, vulnerable=vul, dealer=dealer)
        self.gs.new_deal()
        return self.gs.observation(self.gs.next_actor())

    def step(self):
        """Advances the game by one action and returns the transition tuple.

        Determines which player is next to act, fetches that player's action
        (via ``bid()`` during the auction phase or ``play_card()`` during the
        play phase), applies it to the game state, then assembles the return
        values.

        If the game is already in ``COMPLETE`` phase when ``step()`` is
        called, it returns immediately with ``(None, 0, True)`` without
        querying any player.

        Returns:
            A three-tuple ``(observation, reward, done)`` where:

            - ``observation``: The observation dict for the *next* active
              player, or ``None`` when the board is complete.
            - ``reward``: ``(score_ns, score_ew)`` tuple of ints when
              ``done=True``; ``(0, 0)`` on all intermediate steps.
            - ``done``: ``True`` when the board has reached the ``COMPLETE``
              phase, ``False`` otherwise.
        """
        gs = self.gs
        if gs.phase not in ('AUCTION', 'PLAY'):
            return None, 0, True
        actor = gs.next_actor()
        obs = gs.observation(actor)
        if gs.phase == 'AUCTION':
            action = self.players[actor].bid(obs)
            gs.apply_call(action)
        else:
            action = self.players[actor].play_card(obs)
            gs.play_card(actor, action)
        done = gs.phase == 'COMPLETE'
        reward = (gs.score_ns, gs.score_ew) if done else (0, 0)
        next_actor = gs.next_actor() if not done else None
        return gs.observation(next_actor) if next_actor else None, reward, done

    def run_episode(self):
        """Convenience method that runs a full board and returns the final reward.

        Calls :meth:`reset` to start a new board, then loops :meth:`step`
        until the board is complete, discarding intermediate observations and
        rewards.

        Returns:
            A ``(score_ns, score_ew)`` tuple of ints representing the final
            bridge score for the completed board.
        """
        self.reset()
        while True:
            _, reward, done = self.step()
            if done: return reward
