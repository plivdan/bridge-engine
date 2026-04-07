# Architecture Deep Dive

## Package Layout

```
engine/       Core bridge rules — no AI logic, no external dependencies
ai/           AI agents — depends on engine/
empirical/    Data collection and optimization — depends on engine/ and ai/
tests/        Test suites — depends on all packages
```

Import direction: `engine/ <- ai/ <- empirical/`. The engine has zero knowledge of AI agents or optimization. AI agents import engine primitives. Empirical tools import both.

## Data Flow

A complete board flows through five stages:

```
deal() --> GameState.new_deal()
               |
               v
           AuctionState
           |-- apply_call() x N    (players bid in turn)
           |-- valid_calls()       (legal bid enumeration)
           +-- result()            (contract, declarer, doubled)
               |
               v
           PlayState
           |-- play_card() x 52   (4 cards x 13 tricks)
           |-- valid_cards()       (follow-suit enforcement)
           +-- result()            (tricks_ns, tricks_ew, declarer_tricks)
               |
               v
           score(contract, declarer, doubled, tricks, vulnerable)
               |
               v
           GameState.score_ns / score_ew
```

### Initialization

`deal()` creates a standard 52-card deck, shuffles it, and partitions into four sorted hands of 13. Hands are keyed by seat index (0=N, 1=E, 2=S, 3=W).

`GameState.new_deal()` stores the hands, creates an `AuctionState` with the current dealer, and transitions to the AUCTION phase.

### Auction

`AuctionState` maintains the list of all calls and a `current_player` pointer that advances clockwise (mod 4) after each call.

`valid_calls()` computes legality:
- PASS is always valid
- Any bid strictly higher than the last bid (by level, then strain C < D < H < S < NT)
- DOUBLE: only if the last bid was by the opposing side and not already doubled
- REDOUBLE: only if the last action was a double by the opposing side

**Declarer assignment** (`apply_call`): when a normal bid is made, the engine scans all prior calls to find the **first player on the same partnership** who bid the same strain. That player becomes declarer.

The auction ends when three consecutive passes follow a bid, or when four passes occur from the start (passed out).

### Play

`PlayState` is initialized with:
- `hands`: mutable copy of all four hands
- `trump`: the contract strain (NT stored as `Suit.NT`, converted to `None` for trick evaluation)
- `declarer`, `dummy` (= declarer + 2 mod 4), `leader` (= declarer + 1 mod 4)

A `Trick` collects cards from four seats in clockwise order from the leader. Follow-suit is enforced: if the seat holds any card in the led suit, only those cards are valid.

## The `current_seat` / `current_player` Split

This is the most important design decision in the engine.

### Why two concepts?

In bridge, the declarer physically plays cards from both their own hand and dummy's hand. The defenders play only their own cards. This creates an asymmetry:

- **`current_seat`**: the physical position around the table whose turn it is. Always advances clockwise.
- **`current_player`**: the human/agent who controls that seat. Equals `current_seat` for all positions **except** dummy, where it equals `declarer`.

### Worked Example

Setup: North is declarer, South is dummy, East leads.

```
Trick 1:
  Seat  | current_seat | current_player | Who acts?
  ------+--------------+----------------+----------
  East  |      1       |       1        | East (defender)
  South |      2       |       0        | North (declarer plays for dummy)
  West  |      3       |       3        | West (defender)
  North |      0       |       0        | North (declarer plays own hand)
```

When `current_seat = 2` (South/dummy):
- `current_player` returns `0` (North/declarer)
- `play_card(actor=0, card)` is called
- The card is validated against and removed from seat 2's (dummy's) hand

### Code path through `play_card`

```python
def play_card(self, actor, card):
    seat = self.current_seat                          # physical seat
    expected = self.declarer if seat == self.dummy else seat
    assert actor == expected                           # validate
    assert card in self.valid_cards(seat)              # follow-suit on SEAT
    self.hands[seat].remove(card)                      # remove from SEAT's hand
    self.current_trick.add_card(seat, card)
```

## AI Agent Architecture

### Bidding Agent (`ai/bidding_agent.py`)

State-machine bidder implementing Standard American Yellow Card. Recomputes its phase from the auction history on every `bid()` call (stateless design).

```
BidPhase enum:
  OPENING -> RESPONDING -> OPENER_REBID -> RESPONDER_REBID -> SIGNOFF
                                        -> SLAM_INVESTIGATION
  OVERCALL -> COMPETITIVE
```

Key methods:
- `_determine_phase()`: scans auction history to classify the current situation
- `_estimate_partner()`: infers partner's HCP range from their bids
- `_target_level()`: computes how high to bid based on combined strength
- All thresholds come from `BridgeParams` (no hardcoded magic numbers)

### Card Play Agent (`ai/cardplay_agent.py`)

Card-counting play agent with separate logic for opening lead, declarer play, and defense.

```
CardTracker: played cards, voids, high card mastery
DeclarerPlan: winners, shortfall, finesse suits, ruff potential

Opening lead: partner suit > sequence > 4th best (NT) / singleton (suit)
Declarer:     ruff setup > finesse > cash winners > draw trumps
Defense:      3rd hand high, 2nd hand low, cover honor, ruff when void
```

Optional Monte Carlo mode samples random opponent hands and evaluates cards via greedy playout.

### Parameter System (`ai/bridge_params.py`)

49 tunable parameters in a frozen-compatible dataclass:
- Opening thresholds (min HCP, NT ranges, strong 2C)
- Response thresholds (min HCP, raise levels)
- Combined targets (game, slam, invitational)
- Partner estimation fraction
- Rebid brackets, overcall ranges
- Hand evaluation weights (distribution, support points)
- Card play tactics (cover honor rank, sequence rank, trump draw rank)
- Monte Carlo settings

`BridgeParams()` with defaults reproduces exact pre-parameterization behavior. `to_json()`/`from_json()` for persistence.

## Empirical Optimization

### Data Collection (`empirical/bridge_stats.py`)

`collect_boards()` runs N boards silently, recording 30+ features per board in `BoardRecord` dataclass: per-seat HCP/TP/LTC/QT, partnership fit lengths, contract outcome, tricks, score.

### EV Tables (`empirical/bridge_tables.py`)

Built from mass simulation data:
- **Game EV**: average score by (combined_hcp_bin, vulnerability) for game vs part-score
- **Slam EV**: slam vs game EV by (combined_hcp_bin, has_fit, vulnerability)
- **Make rates**: P(make | hcp_bin, level, vulnerability)
- **Partner distributions**: empirical P(partner_hcp | bid_category)

### Optimizer (`empirical/bridge_optimize.py`)

Coordinate descent: sweeps each parameter over a candidate range, keeping the value that maximizes per-board score advantage. Two phases: high-impact params (8 params), then medium-impact (10 params). ~10 minutes for a full run.

## The Observation Dict

`GameState.observation(player)` is the primary interface for ML agents.

### Always present
| Key | Type | Description |
|-----|------|-------------|
| `board_num` | int | Board number |
| `vulnerable` | dict | `{'NS': bool, 'EW': bool}` |
| `dealer` | int | Seat index of dealer |
| `phase` | str | `'DEAL'`, `'AUCTION'`, `'PLAY'`, or `'COMPLETE'` |
| `player` | int | The seat index this observation is for |
| `hand` | list[Card] | The requesting player's current hand |

### During auction (adds)
| Key | Type | Description |
|-----|------|-------------|
| `calls` | list[Bid] | All bids so far |
| `current_bidder` | int | Seat index of next bidder |
| `valid_calls` | list[Bid] | Legal bids for current bidder |
| `contract` | Bid or None | Current highest bid |
| `declarer` | int or None | Current declarer assignment |

### During play (adds)
| Key | Type | Description |
|-----|------|-------------|
| `dummy_hand` | list[Card] | Dummy's remaining cards (visible to all) |
| `tricks_ns` | int | Tricks won by N-S |
| `tricks_ew` | int | Tricks won by E-W |
| `completed_tricks` | list[Trick] | All finished tricks |
| `current_trick` | Trick | Trick in progress |
| `trump` | Suit | Contract strain |
| `current_player` | int | Actor who should play next |
| `current_seat` | int | Physical seat that plays next |
| `valid_cards` | list[Card] | Legal cards for `current_seat` |

## Extension Points

### Neural network player
1. Subclass `Player` from `engine.player`
2. In `bid()`: encode observation -> tensor -> policy over `valid_calls`
3. In `play_card()`: encode observation -> tensor -> policy over `valid_cards`
4. Mask invalid actions using the valid lists

### Adding bidding conventions
Extend `StateMachineBidder` in `ai/bidding_agent.py` with new phase handlers for Stayman, Jacoby transfers, etc.

### Double-dummy analysis
`PlayState` exposes all four hands — a minimax solver can enumerate all legal plays via `valid_cards(seat)` and recurse through cloned states.
