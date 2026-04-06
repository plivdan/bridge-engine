# Bridge Engine

A complete duplicate contract bridge engine in Python, designed for both human-readable simulation and ML/RL research. Covers the full lifecycle of a bridge board — dealing, auction (bidding), card play with follow-suit enforcement, and duplicate scoring — plus a gym-style self-play environment for training agents.

## Architecture

```
card.py          Suit, Rank, Card, DECK, deal()
   │
   ├──► auction.py    Bid, AuctionState (bidding logic, declarer assignment)
   ├──► play.py       Trick, PlayState (card play, follow-suit, dummy control)
   └──► scoring.py    score(), score_rubber() (duplicate & rubber scoring)
            │
            ▼
        state.py      GameState (orchestrates DEAL → AUCTION → PLAY → COMPLETE)
            │
            ▼
        game.py       Game (multi-board sessions), SelfPlayEnv (RL interface)
            │
            ▼
        player.py     Player ABC, RandomPlayer, PassingPlayer,
                      SimpleHeuristicPlayer, RuleBasedPlayer
            │
            ▼
        main.py       Demo entry point
```

## Module Reference

### `card.py`
Core primitives. `Suit` (C/D/H/S/NT) and `Rank` (TWO–ACE) are IntEnums. `Card` is a frozen, ordered dataclass with `.hcp()` returning Milton Work points (A=4, K=3, Q=2, J=1). `DECK` holds all 52 cards. `deal()` shuffles and returns `{0: [...13 cards...], 1: ..., 2: ..., 3: ...}`.

### `auction.py`
`Bid(level, strain, special)` — frozen dataclass with natural ordering by `(level, strain)`. Special calls: `PASS`, `DOUBLE`, `REDOUBLE`. `AuctionState` tracks all calls, validates legality (`valid_calls()`), applies bids (`apply_call()`), and assigns the declarer as the **first player on the winning side to bid the contract strain**.

### `play.py`
`Trick` collects four cards and determines the winner (led-suit priority, trump override). `PlayState` manages 13 tricks with follow-suit enforcement. Key design: **`current_seat`** is the physical position that must play; **`current_player`** is the actor who controls that seat (equals declarer when dummy is seated). `play_card(actor, card)` validates the actor and removes the card from `current_seat`'s hand.

### `scoring.py`
`score(contract, declarer, doubled, tricks_made, vulnerable)` returns the duplicate score — positive if the contract makes, negative for undertricks. `doubled` is 0/1/2. Handles game/slam bonuses, doubled undertrick schedules, and vulnerability. `score_rubber()` provides rubber bridge below/above-the-line scoring.

### `state.py`
`GameState` orchestrates a single board through four phases: DEAL → AUCTION → PLAY → COMPLETE. `observation(player)` returns an ML-ready dict with hand, auction history, valid actions, trick state, and dummy's hand. `next_actor()` returns the player who should act next (declarer when dummy is seated).

### `game.py`
`Game(players, num_boards)` runs a batch of boards with the standard 16-board vulnerability cycle. `SelfPlayEnv(players)` provides a gym-style `reset()`/`step()` interface for RL training — rewards are `(score_ns, score_ew)` tuples on terminal steps.

### `player.py`
Abstract `Player` base class with `bid(obs)` and `play_card(obs)`. Four implementations:
- **RandomPlayer** — uniform random from valid options (baseline)
- **PassingPlayer** — always passes, random card play
- **SimpleHeuristicPlayer** — opens 12+ HCP, responds 6+ HCP, plays high cards first
- **RuleBasedPlayer** — heuristic bidding + smarter play (sequences, cheapest winner, partner awareness)

## Key Invariants

- **Declarer controls dummy**: `current_player` returns the declarer when `current_seat` is dummy. All `play_card()` calls for dummy use the declarer as actor.
- **Scoring sign**: positive = declarer's side scores; negative = penalty to declarer's side. Only one side scores per board.
- **Vulnerability**: follows the standard 16-board duplicate cycle in `VUL_SCHEDULE`.
- **Dealer rotation**: `(board_num - 1) % 4` → N, E, S, W.
- **13 tricks**: every completed board has exactly 13 tricks, all hands empty.

## Quickstart

### Run the demo
```bash
python main.py
```
Runs games with Random, Heuristic, RuleBased players and a SelfPlayEnv loop.

### Run tests
```bash
python test_bridge.py
```
446 tests covering cards, bidding, play, scoring, game invariants, and stress tests.

### Implement a custom AI player
```python
from player import Player
from card import Card
from auction import Bid, PASS

class MyPlayer(Player):
    def bid(self, obs):
        """obs has 'hand', 'valid_calls', 'calls', 'dealer', 'player'."""
        valid = obs['valid_calls']
        # Your bidding logic here
        return PASS

    def play_card(self, obs):
        """obs has 'valid_cards', 'current_trick', 'trump', 'declarer', 'dummy_hand'."""
        valid = obs['valid_cards']
        # Your card play logic here
        return valid[0]
```

Then plug it in:
```python
from game import Game
players = [MyPlayer(i) for i in range(4)]
results = Game(players, num_boards=100).run()
```

## ML / RL Interface

### SelfPlayEnv

```python
from game import SelfPlayEnv
from player import RuleBasedPlayer

players = [RuleBasedPlayer(i) for i in range(4)]
env = SelfPlayEnv(players)

for episode in range(1000):
    obs = env.reset()          # starts a new board
    done = False
    while not done:
        obs, reward, done = env.step()  # player acts internally
    score_ns, score_ew = reward
```

### Observation dict keys

**During auction**: `board_num`, `vulnerable`, `dealer`, `phase`, `player`, `hand`, `calls`, `current_bidder`, `valid_calls`, `contract`, `declarer`

**During play** (adds): `dummy_hand`, `tricks_ns`, `tricks_ew`, `completed_tricks`, `current_trick`, `trump`, `current_player`, `current_seat`, `valid_cards`

### Action spaces
- **Bidding**: one of `valid_calls` — up to 38 options (35 bids + PASS + DOUBLE + REDOUBLE)
- **Play**: one of `valid_cards` — 1 to 13 cards depending on hand size and follow-suit

## Scoring Reference

| Contract | NV Made | V Made | NV -1 | V -1 | NV -1X | V -1X |
|----------|---------|--------|-------|------|--------|-------|
| 1m       | 70      | 70     | -50   | -100 | -100   | -200  |
| 1M       | 80      | 80     | -50   | -100 | -100   | -200  |
| 1NT      | 90      | 90     | -50   | -100 | -100   | -200  |
| 3NT      | 400     | 600    | -50   | -100 | -100   | -200  |
| 4M       | 420     | 620    | -50   | -100 | -100   | -200  |
| 5m       | 400     | 600    | -50   | -100 | -100   | -200  |
| 6NT      | 990     | 1440   | -50   | -100 | -100   | -200  |
| 7NT      | 1520    | 2220   | -50   | -100 | -100   | -200  |

m = minor (clubs/diamonds), M = major (hearts/spades). Overtrick/undertrick details scale — see `scoring.py` for the full formula.
