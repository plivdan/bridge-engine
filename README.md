# Bridge Engine

A complete duplicate contract bridge engine in Python with AI agents, designed for both human-readable simulation and ML/RL research. Covers dealing, auction (bidding), card play with follow-suit enforcement, duplicate scoring, and data-driven parameter optimization.

## Project Structure

```
bridge/
├── engine/              Core bridge engine
│   ├── card.py          Suit, Rank, Card, DECK, deal()
│   ├── auction.py       Bid, AuctionState (bidding logic, declarer assignment)
│   ├── play.py          Trick, PlayState (card play, follow-suit, dummy control)
│   ├── scoring.py       score(), score_rubber() (duplicate & rubber scoring)
│   ├── state.py         GameState (DEAL -> AUCTION -> PLAY -> COMPLETE)
│   ├── game.py          Game (multi-board sessions), SelfPlayEnv (RL interface)
│   └── player.py        Player ABC, RandomPlayer, RuleBasedPlayer, etc.
│
├── ai/                  AI agents and evaluation
│   ├── hand_eval.py     HCP, shape, distribution/support points, LTC, quick tricks
│   ├── bidding_agent.py State-machine bidder (Standard American system)
│   ├── cardplay_agent.py Card play agent with tracking, finesses, Monte Carlo
│   ├── smart_player.py  SmartPlayer — drop-in Player combining both agents
│   └── bridge_params.py BridgeParams dataclass (49 tunable parameters + JSON I/O)
│
├── empirical/           Data-driven optimization
│   ├── bridge_stats.py  Mass simulation -> BoardRecord (50K boards)
│   ├── bridge_tables.py EV tables, make rates, Bayesian partner distributions
│   └── bridge_optimize.py Coordinate descent parameter optimizer
│
├── tests/               Test suites (544 tests total)
│   ├── test_bridge.py   Engine tests (446): cards, auction, play, scoring, invariants
│   ├── test_bidding.py  Bidding agent tests (23): openings, responses, auctions
│   ├── test_cardplay.py Card play tests (14): leads, defense, tracking
│   ├── test_integration.py Head-to-head benchmarks (12): crash resistance, scoring
│   └── test_empirical.py Empirical framework tests (49): params, data, tables, MC
│
├── config/              Configuration
│   └── params_optimized.json  Optimized parameters from coordinate descent
│
└── main.py              Demo entry point
```

## Quickstart

### Run the demo
```bash
python main.py
```

### Run all tests
```bash
python tests/test_bridge.py
python tests/test_bidding.py
python tests/test_cardplay.py
python tests/test_integration.py
python tests/test_empirical.py
```

### Play games with the AI
```python
from ai.smart_player import SmartPlayer
from engine.player import RuleBasedPlayer
from engine.game import Game

# SmartPlayer (NS) vs RuleBasedPlayer (EW)
players = [
    SmartPlayer(0), RuleBasedPlayer(1),
    SmartPlayer(2), RuleBasedPlayer(3),
]
results = Game(players, num_boards=100).run()
```

### Use optimized parameters
```python
from ai.bridge_params import BridgeParams
from ai.smart_player import SmartPlayer

params = BridgeParams.from_json('config/params_optimized.json')
player = SmartPlayer(0, params=params)
```

### Optimize parameters yourself
```bash
python -m empirical.bridge_optimize
```

## Engine Overview

### Core Primitives (`engine/`)

**card.py** — `Suit` (C/D/H/S/NT) and `Rank` (TWO-ACE) as IntEnums. `Card` is a frozen, ordered dataclass with `.hcp()` returning Milton Work points (A=4, K=3, Q=2, J=1). `deal()` shuffles and returns four hands of 13.

**auction.py** — `Bid(level, strain, special)` with natural ordering. `AuctionState` tracks calls, validates legality, and assigns the declarer as the first player on the winning side to bid the contract strain.

**play.py** — `Trick` collects four cards and determines the winner. `PlayState` manages 13 tricks with follow-suit enforcement. Key design: `current_seat` is the physical position; `current_player` is the actor (declarer controls dummy).

**scoring.py** — Full duplicate scoring: game/slam bonuses, doubled undertrick schedules, vulnerability.

**state.py** — `GameState` orchestrates DEAL -> AUCTION -> PLAY -> COMPLETE. `observation(player)` returns an ML-ready dict.

**game.py** — `Game(players, num_boards)` runs batches with the 16-board vulnerability cycle. `SelfPlayEnv` provides a gym-style `reset()`/`step()` RL interface.

**player.py** — Abstract `Player` base with four implementations: `RandomPlayer`, `PassingPlayer`, `SimpleHeuristicPlayer`, `RuleBasedPlayer`.

### AI Agents (`ai/`)

**SmartPlayer** combines a state-machine bidder (Standard American Yellow Card) with a card-counting play agent. It supports:
- Full bidding system: openings (1NT 15-17, 2NT 20-21, 2C 22+, 1-suit 12-21), responses, rebids, overcalls, competitive doubles, Blackwood slam investigation
- Card play: opening leads (sequences, 4th best), declarer planning (finesses, ruffs), defense (3rd hand high, cover honor)
- Card tracking: played cards, void inference, high card mastery
- Optional Monte Carlo trick estimation

**BridgeParams** — 49 tunable parameters covering every bidding threshold, hand evaluation weight, and card play tactic. Defaults reproduce traditional bridge wisdom. JSON serialization for persistence.

### Empirical Optimization (`empirical/`)

**Data collection** — `collect_boards()` runs mass simulations, recording 30+ features per board (HCP, shape, fit length, contract, tricks, score).

**EV tables** — Empirical game/slam expected value by HCP bin and vulnerability. Replaces hardcoded "bid game at 26" with data-driven thresholds.

**Coordinate descent** — Optimizes parameters by playing thousands of boards per configuration. Key findings from optimization:
- Slams disabled (6.9% success rate is unprofitable)
- Opening threshold raised to 13 HCP
- Response threshold raised to 8 HCP
- 1NT range widened to 14-18

## ML / RL Interface

### SelfPlayEnv

```python
from engine.game import SelfPlayEnv
from engine.player import RuleBasedPlayer

players = [RuleBasedPlayer(i) for i in range(4)]
env = SelfPlayEnv(players)

for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done = env.step()
    score_ns, score_ew = reward
```

### Observation dict

**During auction**: `board_num`, `vulnerable`, `dealer`, `phase`, `player`, `hand`, `calls`, `current_bidder`, `valid_calls`, `contract`, `declarer`

**During play** (adds): `dummy_hand`, `tricks_ns`, `tricks_ew`, `completed_tricks`, `current_trick`, `trump`, `current_player`, `current_seat`, `valid_cards`

### Implement a custom player

```python
from engine.player import Player
from engine.auction import PASS

class MyPlayer(Player):
    def bid(self, obs):
        return PASS  # your bidding logic

    def play_card(self, obs):
        return obs['valid_cards'][0]  # your play logic
```

## Key Invariants

- **Declarer controls dummy**: `current_player` returns the declarer when `current_seat` is dummy
- **Scoring sign**: positive = declarer's side scores
- **Vulnerability**: standard 16-board duplicate cycle
- **13 tricks**: every completed board has exactly 13 tricks, all hands empty

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
