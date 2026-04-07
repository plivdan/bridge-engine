# Bridge Engine

A duplicate contract bridge engine in Python with state-machine AI agents, double-dummy analysis, and data-driven parameter optimization. Covers the full lifecycle: dealing, auction, card play, and duplicate scoring.

**SmartPlayer scores +93/board vs RuleBasedPlayer** over 500-board matches with optimized parameters.

## Project Structure

```
bridge/
├── engine/              Core bridge engine
│   ├── card.py          Suit, Rank, Card, DECK, deal()
│   ├── auction.py       Bid, AuctionState (bidding, declarer assignment)
│   ├── play.py          Trick, PlayState (card play, follow-suit, dummy control)
│   ├── scoring.py       score(), score_rubber() (duplicate & rubber scoring)
│   ├── state.py         GameState (DEAL -> AUCTION -> PLAY -> COMPLETE)
│   ├── game.py          Game, SelfPlayEnv (RL interface)
│   ├── player.py        Player ABC, RandomPlayer, RuleBasedPlayer
│   └── dds.py           Double-dummy solver (alpha-beta minimax)
│
├── ai/                  AI agents and evaluation
│   ├── hand_eval.py     HCP, shape, distribution/support points, LTC, quick tricks
│   ├── bidding_agent.py State-machine bidder (Standard American)
│   ├── cardplay_agent.py Card play with tracking, finesses, overruff prevention
│   ├── smart_player.py  SmartPlayer — drop-in Player combining both agents
│   ├── bridge_params.py BridgeParams dataclass (50+ tunable parameters)
│   └── trace.py         Decision tracing for debugging AI choices
│
├── empirical/           Data-driven optimization and measurement
│   ├── bridge_stats.py  Mass simulation -> BoardRecord
│   ├── bridge_tables.py EV tables, make rates, Bayesian partner distributions
│   ├── bridge_optimize.py Coordinate descent parameter optimizer
│   ├── dd_benchmark.py  Benchmark AI vs double-dummy optimal
│   └── report.py        Performance comparison reports
│
├── tests/               Test suites (577 tests)
│   ├── test_bridge.py   Engine (446): cards, auction, play, scoring, invariants
│   ├── test_bidding.py  Bidding (24): openings, responses, auctions, hand eval
│   ├── test_cardplay.py Card play (26): leads, defense, tracking, plan, ruffs
│   ├── test_integration.py Benchmarks (12): head-to-head, crash resistance
│   ├── test_empirical.py Empirical (49): params, data collection, tables, MC
│   └── test_dds.py      DD solver (19): known positions, performance
│
├── config/
│   └── params_optimized.json  Optimized parameters from coordinate descent
│
└── main.py              Demo entry point
```

## Quickstart

```bash
# Run the demo
python main.py

# Run all tests
python tests/test_bridge.py && python tests/test_bidding.py && \
python tests/test_cardplay.py && python tests/test_integration.py && \
python tests/test_empirical.py && python tests/test_dds.py

# Play SmartPlayer vs RuleBasedPlayer
python -c "
from ai.smart_player import SmartPlayer
from engine.player import RuleBasedPlayer
from engine.game import Game
results = Game([SmartPlayer(0), RuleBasedPlayer(1),
                SmartPlayer(2), RuleBasedPlayer(3)], num_boards=100).run()
ns = sum(r['score_ns'] for r in results)
ew = sum(r['score_ew'] for r in results)
print(f'NS: {ns}  EW: {ew}  Net: {ns-ew:+d}')
"

# Run performance report
python -m empirical.report

# Optimize parameters (~5 minutes)
python -m empirical.bridge_optimize

# Run DD benchmark
python -m empirical.dd_benchmark
```

## AI System

### SmartPlayer

Drop-in `Player` subclass combining a state-machine bidder with a card-counting play agent.

**Bidding** (Standard American Yellow Card):
- Openings: 1NT (16-17), 2NT (20-21), 2C (22+), 1-suit (13+)
- Responses calibrated by partner estimation with vulnerability awareness
- Opener/responder rebids with fit detection and support point revaluation
- Overcalls, competitive doubles, Blackwood slam investigation
- Won't overbid game unless slam values exist

**Card Play:**
- Opening leads: partner's suit, honor sequences, 4th best (NT), singleton (suit)
- Declarer: dynamic plan (recomputed each trick), positional finesse detection, ruff setup with overruff prevention, smart trump management
- Defense: 3rd hand high, 2nd hand low, cover honor, ruff when void
- CardTracker: played cards, void inference, honor placement, suit split estimation
- Optional Monte Carlo trick estimation

**Tunable Parameters** (`BridgeParams`):
- 50+ parameters covering bidding thresholds, hand evaluation weights, card play tactics
- JSON serialization for persistence
- Coordinate descent optimizer finds optimal values in ~5 minutes
- Vulnerability-aware adjustments (game threshold shifts by vul status)

### Decision Tracing

Enable `trace_enabled=True` in BridgeParams to see every decision:

```python
from ai.bridge_params import BridgeParams
from ai.smart_player import SmartPlayer

params = BridgeParams(trace_enabled=True)
player = SmartPlayer(0, params=params)
# After playing, inspect player.trace_log.summary()
```

### Double-Dummy Solver

Alpha-beta minimax solver with bit-packed hands, transposition table, and equivalent card canonicalization:

```python
from engine.dds import double_dummy_tricks
from engine.card import deal, Suit

hands = deal()
tricks = double_dummy_tricks(hands, Suit.S, declarer=0)
print(f"North can take {tricks} tricks with spades as trump")
```

Exact solve for positions with 8 or fewer cards per hand. Quick-tricks heuristic for full 13-card hands.

## Engine Overview

**card.py** — `Suit` (C/D/H/S/NT) and `Rank` (TWO-ACE) as IntEnums. `Card` is a frozen dataclass with `.hcp()`. `deal()` returns four sorted hands of 13.

**auction.py** — `AuctionState` tracks calls, validates legality, assigns declarer as the first player on the winning side to bid the contract strain.

**play.py** — `PlayState` manages 13 tricks with follow-suit enforcement. `current_seat` is the physical position; `current_player` is the actor (declarer controls dummy).

**scoring.py** — Full duplicate scoring: game/slam bonuses, doubled undertrick schedules, vulnerability.

**state.py** — `GameState` orchestrates DEAL -> AUCTION -> PLAY -> COMPLETE. `observation(player)` returns an ML-ready dict with current (not dealt) hands during play.

**game.py** — `Game(players, num_boards)` with 16-board vulnerability cycle. `SelfPlayEnv` for RL training loops.

## ML / RL Interface

```python
from engine.game import SelfPlayEnv
from engine.player import RuleBasedPlayer

env = SelfPlayEnv([RuleBasedPlayer(i) for i in range(4)])
for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        obs, reward, done = env.step()
    score_ns, score_ew = reward
```

### Observation keys

**Auction**: `board_num`, `vulnerable`, `dealer`, `phase`, `player`, `hand`, `calls`, `current_bidder`, `valid_calls`, `contract`, `declarer`

**Play** (adds): `dummy_hand`, `tricks_ns`, `tricks_ew`, `completed_tricks`, `current_trick`, `trump`, `current_player`, `current_seat`, `valid_cards`

### Custom player

```python
from engine.player import Player
from engine.auction import PASS

class MyPlayer(Player):
    def bid(self, obs):
        return PASS

    def play_card(self, obs):
        return obs['valid_cards'][0]
```

## Key Invariants

- **Declarer controls dummy**: `current_player` returns declarer when `current_seat` is dummy
- **Scoring sign**: positive = declarer's side scores
- **Vulnerability**: standard 16-board duplicate cycle
- **13 tricks**: every completed board uses all 52 cards
- **Observation hands**: during play, `hand` and `dummy_hand` reflect current remaining cards

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
