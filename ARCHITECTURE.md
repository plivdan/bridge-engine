# Architecture Deep Dive

## Data Flow

A complete board flows through five stages:

```
deal() ──► GameState.new_deal()
               │
               ▼
           AuctionState
           ├── apply_call() × N    (players bid in turn)
           ├── valid_calls()       (legal bid enumeration)
           └── result()            (contract, declarer, doubled)
               │
               ▼
           PlayState
           ├── play_card() × 52   (4 cards × 13 tricks)
           ├── valid_cards()       (follow-suit enforcement)
           └── result()            (tricks_ns, tricks_ew, declarer_tricks)
               │
               ▼
           score(contract, declarer, doubled, tricks, vulnerable)
               │
               ▼
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

**Declarer assignment** (`apply_call`): when a normal bid is made, the engine scans all prior calls to find the **first player on the same partnership** who bid the same strain. That player becomes declarer. This implements the bridge rule that declarer is the first player on the winning side to have named the denomination of the final contract.

The auction ends when three consecutive passes follow a bid, or when four passes occur from the start (passed out).

### Play

`PlayState` is initialized with:
- `hands`: mutable copy of all four hands
- `trump`: the contract strain (NT stored as `Suit.NT`, converted to `None` for trick evaluation)
- `declarer`, `dummy` (= declarer + 2 mod 4), `leader` (= declarer + 1 mod 4)

A `Trick` collects cards from four seats in clockwise order from the leader. Follow-suit is enforced: if the seat holds any card in the led suit, only those cards are valid.

## The `current_seat` / `current_player` Split

This is the most important design decision in the engine, and the one most likely to cause confusion.

### Why two concepts?

In bridge, the declarer physically plays cards from both their own hand and dummy's hand. The defenders play only their own cards. This creates an asymmetry:

- **`current_seat`**: the physical position around the table whose turn it is. Always advances clockwise: leader → leader+1 → leader+2 → leader+3.
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
- `play_card(actor=0, card)` is called — the actor is the declarer
- The card is validated against and removed from seat 2's (dummy's) hand
- `valid_cards(2)` returns dummy's legal plays

This means:
1. The RL agent controlling North makes **two decisions** per trick (own hand + dummy)
2. `observation()` always shows `valid_cards` for `current_seat`, not the requesting player
3. External code should always use `next_actor()` to determine who acts, never assume from the seat

### Code path through `play_card`

```python
def play_card(self, actor, card):
    seat = self.current_seat                          # physical seat
    expected = self.declarer if seat == self.dummy else seat  # who should act
    assert actor == expected                           # validate
    assert card in self.valid_cards(seat)              # follow-suit check on SEAT
    self.hands[seat].remove(card)                      # remove from SEAT's hand
    self.current_trick.add_card(seat, card)            # seat plays the card
```

## The Observation Dict

`GameState.observation(player)` is the primary interface for ML agents. It returns a dict with:

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
| `calls` | list[Bid] | All bids so far, in order |
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

**Design note**: `valid_cards` reflects `current_seat`'s legal plays, not the requesting `player`'s. When dummy is seated, `valid_cards` shows dummy's options even if the observation is requested by a different player. This is intentional — the RL agent that is `current_player` needs to know what it can play.

## The SelfPlayEnv Loop

```python
class SelfPlayEnv:
    def reset(self):
        # Increment board, look up vulnerability/dealer, deal, return first obs
        
    def step(self):
        # 1. Get current actor via gs.next_actor()
        # 2. Get observation via gs.observation(actor)
        # 3. If AUCTION: player.bid(obs) → gs.apply_call(action)
        #    If PLAY:    player.play_card(obs) → gs.play_card(actor, action)
        # 4. Return (next_obs, reward, done)
        #    reward = (score_ns, score_ew) if done, else (0, 0)
```

Each `step()` call advances the game by exactly one action. The environment internally queries the appropriate player object. To train a neural network agent, replace one or more `Player` instances with your own implementation that reads the observation dict and returns an action.

### Training loop pseudocode

```python
env = SelfPlayEnv([MyNNPlayer(0), RuleBasedPlayer(1),
                   MyNNPlayer(2), RuleBasedPlayer(3)])

for episode in range(num_episodes):
    obs = env.reset()
    transitions = []
    done = False
    while not done:
        obs, reward, done = env.step()
        # MyNNPlayer.play_card() internally stores (state, action) pairs
    
    # reward = (score_ns, score_ew)
    # Use reward to compute policy gradient / TD update for MyNNPlayer
```

## Known Extension Points

### Adding a neural network player
1. Subclass `Player`
2. In `bid()`: encode `obs['hand']`, `obs['calls']`, `obs['vulnerable']` → tensor → policy over `obs['valid_calls']`
3. In `play_card()`: encode `obs['hand']`, `obs['dummy_hand']`, `obs['completed_tricks']`, `obs['current_trick']` → tensor → policy over `obs['valid_cards']`
4. Mask invalid actions using the `valid_calls`/`valid_cards` lists

### Adding conventions or systems
The `SimpleHeuristicPlayer.bid()` method is a flat decision tree — extend it by adding branches for Stayman, Jacoby transfers, etc. The `RuleBasedPlayer.play_card()` method has hooks for defensive signals (attitude, count) in the `_find_sequence` helper.

### Double-dummy analysis
The `PlayState` exposes all four hands in `self.hands` — a minimax solver can enumerate all legal plays via `valid_cards(seat)` and recurse through `play_card` / `_complete_trick` on a cloned state.

### Extending scoring
`score_rubber()` already handles rubber bridge. For IMPs or matchpoints, wrap `score()` output with the standard conversion tables.
