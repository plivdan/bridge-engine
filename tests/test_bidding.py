"""Bidding agent correctness tests.

Tests crafted hands against expected bids and full auction outcomes.
Uses the same check/section framework as test_bridge.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.card import Card, Suit, Rank
from engine.auction import AuctionState, Bid, PASS, DOUBLE, REDOUBLE, make_bid
from ai.hand_eval import hcp, hand_shape, total_points, distribution_points
from ai.bidding_agent import StateMachineBidder

PASS_COUNT = 0
FAIL_COUNT = 0

def check(cond, label):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"  PASS  {label}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL  {label}")

def section(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


# --- helpers ---------------------------------------------------------------

def make_obs(hand, calls=None, dealer=0, player=0, vulnerable=None):
    """Build a minimal observation dict for bidding tests."""
    calls = calls or []
    a = AuctionState(dealer=dealer)
    for c in calls:
        a.apply_call(c)
    return {
        'hand': hand,
        'calls': calls,
        'dealer': dealer,
        'player': player,
        'valid_calls': a.valid_calls(),
        'contract': a.contract,
        'declarer': a.declarer,
        'vulnerable': vulnerable or {'NS': False, 'EW': False},
    }


def bid_of(bidder, obs):
    """Get bid from a StateMachineBidder."""
    return bidder.bid(obs)


# --- hands -----------------------------------------------------------------

# 17 HCP balanced, all suits stopped → 1NT
HAND_1NT = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]

# 13 HCP + 5 spades → 1S
HAND_1S = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.NINE, Suit.S), Card(Rank.SEVEN, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]

# 20 HCP balanced (4-3-3-3) → 2NT
HAND_2NT = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.THREE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.EIGHT, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C),
]

# 22 HCP balanced (4-3-3-3) → 2C (artificial strong)
HAND_2C = [
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.QUEEN, Suit.H), Card(Rank.EIGHT, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.JACK, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.QUEEN, Suit.C), Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C),
]

# 6 HCP, weak → pass
HAND_WEAK = [
    Card(Rank.KING, Suit.S), Card(Rank.SEVEN, Suit.S), Card(Rank.THREE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.FIVE, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.NINE, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.FOUR, Suit.C), Card(Rank.TWO, Suit.C),
]

# 13 HCP, 5 hearts → 1H
HAND_1H = [
    Card(Rank.QUEEN, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H), Card(Rank.NINE, Suit.H),
    Card(Rank.SEVEN, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FIVE, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]

# 12 HCP, 4-4 minors, no 5-card major → 1D (longer minor)
HAND_1D = [
    Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]

# Responding hand: 10 HCP, 4 hearts (for limit raise after partner opens 1H)
HAND_RESP_HEARTS = [
    Card(Rank.QUEEN, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.NINE, Suit.H),
    Card(Rank.SEVEN, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.SIX, Suit.C), Card(Rank.FOUR, Suit.C), Card(Rank.TWO, Suit.C),
]

# Responding hand: 10 HCP, balanced → should bid 3NT over partner's 1NT
HAND_RESP_NT = [
    Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.TWO, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.FIVE, Suit.C),
]

# Weak responding hand: 4 HCP → should pass
HAND_RESP_WEAK = [
    Card(Rank.SEVEN, Suit.S), Card(Rank.FIVE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.NINE, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.SIX, Suit.H),
    Card(Rank.QUEEN, Suit.D), Card(Rank.FOUR, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C),
    Card(Rank.TWO, Suit.C),
]


# ── OPENING BID SELECTION ────────────────────────────────────────
section("OPENING BID SELECTION")

b = StateMachineBidder(0)

obs = make_obs(HAND_1NT)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.NT), f"15 HCP balanced -> 1NT (got {bid})")

obs = make_obs(HAND_1S)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.S), f"13 HCP + 5S -> 1S (got {bid})")

obs = make_obs(HAND_2NT)
bid = b.bid(obs)
check(bid == make_bid(2, Suit.NT), f"20 HCP balanced -> 2NT (got {bid})")

obs = make_obs(HAND_2C)
bid = b.bid(obs)
check(bid == make_bid(2, Suit.C), f"22+ HCP -> 2C (got {bid})")

obs = make_obs(HAND_WEAK)
bid = b.bid(obs)
check(bid == PASS, f"8 HCP weak -> PASS (got {bid})")

obs = make_obs(HAND_1H)
bid = b.bid(obs)
check(bid == make_bid(1, Suit.H), f"13 HCP + 5H -> 1H (got {bid})")

obs = make_obs(HAND_1D)
bid = b.bid(obs)
check(bid.strain in (Suit.D, Suit.C) and bid.level == 1,
      f"12 HCP minors -> 1m (got {bid})")


# ── RESPONDING ────────────────────────────────────────────────────
section("RESPONDING")

# Partner (seat 0) opens 1H, we are seat 2 responding
# N opens 1H, E passes, S responds
resp_bidder = StateMachineBidder(2)

calls_1h = [make_bid(1, Suit.H), PASS]  # N:1H, E:PASS, now S bids
obs = make_obs(HAND_RESP_HEARTS, calls=calls_1h, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid.strain == Suit.H and bid.level >= 2,
      f"10 HCP + 4H, partner opens 1H -> raise hearts (got {bid})")
check(bid.level == 3,
      f"10 HCP + 4H should be limit raise to 3H (got {bid})")

# Partner opens 1NT, we have 10 HCP → 3NT
calls_1nt = [make_bid(1, Suit.NT), PASS]
obs = make_obs(HAND_RESP_NT, calls=calls_1nt, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == make_bid(3, Suit.NT), f"10 HCP over 1NT -> 3NT (got {bid})")

# Partner opens 1H, we have 4 HCP → PASS
calls_1h2 = [make_bid(1, Suit.H), PASS]
obs = make_obs(HAND_RESP_WEAK, calls=calls_1h2, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == PASS, f"4 HCP over 1H -> PASS (got {bid})")

# Partner opens 1NT, we have 4 HCP → PASS
calls_1nt2 = [make_bid(1, Suit.NT), PASS]
obs = make_obs(HAND_RESP_WEAK, calls=calls_1nt2, dealer=0, player=2)
bid = resp_bidder.bid(obs)
check(bid == PASS, f"4 HCP over 1NT -> PASS (got {bid})")


# ── FULL AUCTION: GAME REACHED ───────────────────────────────────
section("FULL AUCTION: GAME REACHED")

import contextlib, io

def run_auction(hands, dealer=0):
    """Run a full auction with SmartBidders and return the result."""
    from engine.state import GameState
    gs = GameState(board_num=1, vulnerable={'NS': False, 'EW': False}, dealer=dealer)
    gs.hands = hands
    gs.auction = AuctionState(dealer=dealer)
    gs.phase = 'AUCTION'
    bidders = {i: StateMachineBidder(i) for i in range(4)}
    with contextlib.redirect_stdout(io.StringIO()):
        while gs.phase == 'AUCTION':
            actor = gs.next_actor()
            obs = gs.observation(actor)
            call = bidders[actor].bid(obs)
            gs.apply_call(call)
    return gs.auction

# NS have 26+ combined points with 8-spade fit → should reach 4S
hand_n_game = [  # 14 HCP + 5S
    Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
    Card(Rank.NINE, Suit.S), Card(Rank.SEVEN, Suit.S),
    Card(Rank.ACE, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.KING, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.TWO, Suit.C),
]
hand_s_game = [  # 13 HCP + 3S support
    Card(Rank.QUEEN, Suit.S), Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.ACE, Suit.D), Card(Rank.QUEEN, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.JACK, Suit.C), Card(Rank.NINE, Suit.C), Card(Rank.SEVEN, Suit.C),
    Card(Rank.TWO, Suit.C),
]
hand_e_weak = [  # 5 HCP
    Card(Rank.EIGHT, Suit.S), Card(Rank.FOUR, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.NINE, Suit.H), Card(Rank.SEVEN, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.TWO, Suit.D),
    Card(Rank.KING, Suit.C), Card(Rank.TEN, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.FOUR, Suit.C), Card(Rank.THREE, Suit.C),
]
hand_w_weak = [  # 5 HCP
    Card(Rank.SIX, Suit.S), Card(Rank.THREE, Suit.S), Card(Rank.TWO, Suit.S),
    Card(Rank.TEN, Suit.H), Card(Rank.EIGHT, Suit.H), Card(Rank.SIX, Suit.H),
    Card(Rank.TEN, Suit.D), Card(Rank.NINE, Suit.D), Card(Rank.SEVEN, Suit.D),
    Card(Rank.SIX, Suit.D), Card(Rank.FIVE, Suit.D),
    Card(Rank.ACE, Suit.C), Card(Rank.EIGHT, Suit.C),
]

auction = run_auction({0: hand_n_game, 1: hand_e_weak, 2: hand_s_game, 3: hand_w_weak})
contract = auction.contract
declarer = auction.declarer
check(contract is not None, "game auction produced a contract")
if contract:
    check(contract.level >= 3,
          f"NS with 27 pts reached level {contract.level} (need 3+)")
    check(declarer in (0, 2),
          f"NS should be declarer (got seat {declarer})")


# NS have ~24 combined → should stay below game
hand_n_part = [  # 12 HCP
    Card(Rank.ACE, Suit.S), Card(Rank.JACK, Suit.S), Card(Rank.NINE, Suit.S),
    Card(Rank.SEVEN, Suit.S),
    Card(Rank.KING, Suit.H), Card(Rank.FIVE, Suit.H), Card(Rank.THREE, Suit.H),
    Card(Rank.QUEEN, Suit.D), Card(Rank.FOUR, Suit.D),
    Card(Rank.EIGHT, Suit.C), Card(Rank.SIX, Suit.C), Card(Rank.FOUR, Suit.C),
    Card(Rank.TWO, Suit.C),
]
hand_s_part = [  # 10 HCP
    Card(Rank.KING, Suit.S), Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.S),
    Card(Rank.QUEEN, Suit.H), Card(Rank.JACK, Suit.H), Card(Rank.FOUR, Suit.H),
    Card(Rank.JACK, Suit.D), Card(Rank.EIGHT, Suit.D), Card(Rank.THREE, Suit.D),
    Card(Rank.NINE, Suit.C), Card(Rank.SEVEN, Suit.C), Card(Rank.FIVE, Suit.C),
    Card(Rank.THREE, Suit.C),
]

auction2 = run_auction({0: hand_n_part, 1: hand_e_weak, 2: hand_s_part, 3: hand_w_weak})
contract2 = auction2.contract
if contract2:
    check(contract2.level <= 3,
          f"NS with ~22 pts should stay low (got {contract2.level}{contract2.strain})")
else:
    check(True, "NS with ~22 pts: passed out is acceptable")


# ── HAND EVALUATION CONSISTENCY ──────────────────────────────────
section("HAND EVALUATION CONSISTENCY")

check(hcp(HAND_1NT) == 17, f"HAND_1NT HCP = {hcp(HAND_1NT)} (expect 17)")
check(hcp(HAND_1S) == 13, f"HAND_1S HCP = {hcp(HAND_1S)} (expect 13)")
check(hcp(HAND_2NT) == 20, f"HAND_2NT HCP = {hcp(HAND_2NT)} (expect 20)")
check(hcp(HAND_2C) == 22, f"HAND_2C HCP = {hcp(HAND_2C)} (expect 22)")
check(hcp(HAND_WEAK) == 6, f"HAND_WEAK HCP = {hcp(HAND_WEAK)} (expect 6)")
check(hand_shape(HAND_1NT).is_balanced, "HAND_1NT is balanced")
check(hand_shape(HAND_1H).is_balanced,
      "HAND_1H 5-3-3-2 is balanced")
check(hand_shape(HAND_1H).longest_suit == Suit.H,
      "HAND_1H longest suit is hearts")


# ── NT-RESPONSE CONVENTIONS (Batch 1) ────────────────────────────
section("NT-RESPONSE CONVENTIONS: STAYMAN / JACOBY TRANSFERS / GERBER")

# Helper: 13-card hand with specified suit lengths and known honors.
def _pad_hand(honors, lengths):
    """Build a 13-card hand from a list of Card honors plus filler spot cards.

    honors: list[Card] — the specific honors you want to place.
    lengths: dict[Suit, int] — total length per suit; filler added to match.
    """
    # Spot cards (2-9) available per suit, skipping any we already placed.
    all_spots = {s: [Rank(r) for r in range(2, 10)] for s in
                 (Suit.S, Suit.H, Suit.D, Suit.C)}
    for c in honors:
        if c.rank in all_spots[c.suit]:
            all_spots[c.suit].remove(c.rank)
    hand = list(honors)
    by_suit = {s: [c for c in honors if c.suit == s] for s in all_spots}
    for s, target_len in lengths.items():
        need = target_len - len(by_suit[s])
        for _ in range(max(0, need)):
            hand.append(Card(all_spots[s].pop(0), s))
    return hand

# Stayman: 9 HCP with 4 hearts, 4-3-3-3 → 2C
# Responder hand: Kx in S, AJxx in H, Qxx in D, Kxx in C (not going to pass it manually; use _pad_hand)
hand_stayman = _pad_hand(
    [Card(Rank.ACE, Suit.H), Card(Rank.JACK, Suit.H),
     Card(Rank.QUEEN, Suit.D), Card(Rank.KING, Suit.C)],
    {Suit.S: 3, Suit.H: 4, Suit.D: 3, Suit.C: 3},
)
check(hcp(hand_stayman) == 10, f"Stayman test hand HCP = {hcp(hand_stayman)} (expect 10)")
check(hand_shape(hand_stayman).length(Suit.H) == 4, "Stayman hand has 4 hearts")

resp = StateMachineBidder(2)
calls_1nt = [make_bid(1, Suit.NT), PASS]
obs = make_obs(hand_stayman, calls=calls_1nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.C),
      f"10 HCP + 4 hearts over 1NT -> 2C Stayman (got {bid})")

# Jacoby Transfer to hearts: 5 hearts, any strength → 2D
hand_transfer_h = _pad_hand(
    [Card(Rank.KING, Suit.H), Card(Rank.QUEEN, Suit.H),
     Card(Rank.JACK, Suit.H)],
    {Suit.S: 3, Suit.H: 5, Suit.D: 3, Suit.C: 2},
)
check(hand_shape(hand_transfer_h).length(Suit.H) == 5, "Transfer-H hand has 5 hearts")
obs = make_obs(hand_transfer_h, calls=calls_1nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.D),
      f"5-card heart over 1NT -> 2D Jacoby Transfer (got {bid})")

# Jacoby Transfer to spades: 5 spades → 2H
hand_transfer_s = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
     Card(Rank.JACK, Suit.S)],
    {Suit.S: 5, Suit.H: 3, Suit.D: 3, Suit.C: 2},
)
obs = make_obs(hand_transfer_s, calls=calls_1nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.H),
      f"5-card spade over 1NT -> 2H Jacoby Transfer (got {bid})")

# Weak Jacoby Transfer: 3 HCP + 5 hearts → still transfer (then pass)
hand_transfer_weak = _pad_hand(
    [Card(Rank.JACK, Suit.H)],
    {Suit.S: 3, Suit.H: 5, Suit.D: 3, Suit.C: 2},
)
check(hcp(hand_transfer_weak) <= 3, "Weak-transfer hand is weak")
obs = make_obs(hand_transfer_weak, calls=calls_1nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.D),
      f"Weak 5-card heart over 1NT -> still transfers (got {bid})")

# Hand with no 4/5 card major and modest values → still 3NT (natural path)
hand_no_major = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.H),
     Card(Rank.ACE, Suit.D), Card(Rank.KING, Suit.D),
     Card(Rank.JACK, Suit.C)],
    {Suit.S: 3, Suit.H: 3, Suit.D: 4, Suit.C: 3},
)
check(hcp(hand_no_major) == 13, f"no-major hand HCP = {hcp(hand_no_major)} (expect 13)")
obs = make_obs(hand_no_major, calls=calls_1nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(3, Suit.NT),
      f"13 HCP, no 4-card major over 1NT -> 3NT (got {bid})")

# Opener completes transfer: 1NT-2D, opener has 2 hearts + 15 HCP → 2H (simple accept)
# Opener hand: 15 HCP, 2 hearts, balanced 4-2-4-3 (no 4-card support, no super-accept)
opener_min = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.S),
     Card(Rank.JACK, Suit.S), Card(Rank.KING, Suit.H),
     Card(Rank.ACE, Suit.D), Card(Rank.QUEEN, Suit.D)],
    {Suit.S: 4, Suit.H: 2, Suit.D: 4, Suit.C: 3},
)
check(hcp(opener_min) == 15, f"opener_min HCP = {hcp(opener_min)} (expect 15)")

# Seat 0 opens 1NT, seat 1 PASS, seat 2 (partner) 2D, seat 3 PASS, seat 0 rebids
opener = StateMachineBidder(0)
calls_after_transfer = [
    make_bid(1, Suit.NT),  # N: 1NT
    PASS,                   # E
    make_bid(2, Suit.D),    # S: 2D transfer
    PASS,                   # W
]
obs = make_obs(opener_min, calls=calls_after_transfer, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(2, Suit.H),
      f"1NT-P-2D-P, opener (15 HCP, 2H) -> 2H accept (got {bid})")

# Super-accept: opener has 17 HCP + 4 hearts → 3H
# AKx S (7) + AKxx H (7) + Qxx D (2) + Jxx C (1) = 17
opener_max = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
     Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H),
     Card(Rank.QUEEN, Suit.D), Card(Rank.JACK, Suit.C)],
    {Suit.S: 3, Suit.H: 4, Suit.D: 3, Suit.C: 3},
)
check(hcp(opener_max) == 17, f"opener_max HCP = {hcp(opener_max)} (expect 17)")
check(hand_shape(opener_max).length(Suit.H) == 4, "opener_max has 4 hearts")
obs = make_obs(opener_max, calls=calls_after_transfer, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(3, Suit.H),
      f"1NT-P-2D-P, opener (17 HCP, 4H) -> 3H super-accept (got {bid})")

# Opener answers Stayman with 4 hearts → 2H
opener_stayman_h = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.H),
     Card(Rank.QUEEN, Suit.H), Card(Rank.ACE, Suit.D),
     Card(Rank.KING, Suit.C), Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 3, Suit.H: 4, Suit.D: 3, Suit.C: 3},
)
calls_after_stayman = [
    make_bid(1, Suit.NT), PASS, make_bid(2, Suit.C), PASS,
]
obs = make_obs(opener_stayman_h, calls=calls_after_stayman, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(2, Suit.H),
      f"1NT-P-2C-P, opener with 4H -> 2H (got {bid})")

# Opener answers Stayman with no 4-card major → 2D denial
opener_stayman_no = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
     Card(Rank.QUEEN, Suit.H), Card(Rank.ACE, Suit.D),
     Card(Rank.KING, Suit.D), Card(Rank.KING, Suit.C)],
    {Suit.S: 3, Suit.H: 3, Suit.D: 4, Suit.C: 3},
)
obs = make_obs(opener_stayman_no, calls=calls_after_stayman, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(2, Suit.D),
      f"1NT-P-2C-P, opener no 4-card major -> 2D denial (got {bid})")

# Responder post-transfer: weak hand PASSES completion (does not raise past 2H)
calls_post_transfer = [
    make_bid(1, Suit.NT), PASS, make_bid(2, Suit.D), PASS,
    make_bid(2, Suit.H),  PASS,
]
obs = make_obs(hand_transfer_weak, calls=calls_post_transfer, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == PASS,
      f"1NT-P-2D-P-2H-P, weak hand -> PASS (got {bid})")

# Responder post-transfer: game values + 5 hearts → 3NT (let opener pick)
# Kx S (3) + AKxxx H (7) + Qxx D (2) + xxx C (0) = 12 HCP
hand_trans_game = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.ACE, Suit.H),
     Card(Rank.KING, Suit.H), Card(Rank.QUEEN, Suit.D)],
    {Suit.S: 2, Suit.H: 5, Suit.D: 3, Suit.C: 3},
)
check(hcp(hand_trans_game) >= 10 and hcp(hand_trans_game) <= 15,
      f"trans_game HCP = {hcp(hand_trans_game)} (expect 10-15)")
obs = make_obs(hand_trans_game, calls=calls_post_transfer, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(3, Suit.NT),
      f"1NT-P-2D-P-2H-P, 5H game values -> 3NT (got {bid})")

# Responder post-transfer: game values + 6 hearts → 4H
# x S (0) + KQJxxx H (6) + Axx D (4) + Jxx C (1) = 11 HCP, 6 hearts
hand_trans_6h = _pad_hand(
    [Card(Rank.KING, Suit.H), Card(Rank.QUEEN, Suit.H),
     Card(Rank.JACK, Suit.H), Card(Rank.ACE, Suit.D),
     Card(Rank.JACK, Suit.C)],
    {Suit.S: 1, Suit.H: 6, Suit.D: 3, Suit.C: 3},
)
check(hand_shape(hand_trans_6h).length(Suit.H) == 6, "trans_6h has 6 hearts")
obs = make_obs(hand_trans_6h, calls=calls_post_transfer, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(4, Suit.H),
      f"1NT-P-2D-P-2H-P, 6H game values -> 4H (got {bid})")

# Full auction: 5-3 heart fit reached via transfer
# N opens 1NT (15-17), S has 10 HCP + 5H → 1NT-2D-2H-3NT-4H path expected
# (or 1NT-2D-2H-...-4H if 5-card transfer + game values)
from engine.state import GameState

hand_n_1nt = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
     Card(Rank.QUEEN, Suit.H), Card(Rank.ACE, Suit.D),
     Card(Rank.JACK, Suit.D), Card(Rank.KING, Suit.C),
     Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 3, Suit.H: 3, Suit.D: 4, Suit.C: 3},
)
hand_s_transfer = _pad_hand(
    [Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H),
     Card(Rank.JACK, Suit.H), Card(Rank.QUEEN, Suit.D),
     Card(Rank.KING, Suit.C)],
    {Suit.S: 3, Suit.H: 5, Suit.D: 3, Suit.C: 2},
)

auction_transfer = run_auction({
    0: hand_n_1nt, 1: hand_e_weak, 2: hand_s_transfer, 3: hand_w_weak,
})
contract_t = auction_transfer.contract
check(contract_t is not None, "transfer auction produced a contract")
if contract_t:
    check(contract_t.strain == Suit.H or contract_t.strain == Suit.NT,
          f"1NT with 5H opposite should reach H or NT (got {contract_t.strain})")
    check(contract_t.level >= 3,
          f"~26 combined should reach game level (got {contract_t.level})")


# ── SLAM MACHINERY (Batch 2) ─────────────────────────────────────
section("SLAM MACHINERY: RKCB / JACOBY 2NT / SPLINTERS")

from ai.bid_meaning import (
    decode_rkcb_response, rkcb_response_for, splinter_short_suit,
    SPLINTER_BIDS,
)

# Jacoby 2NT: 14 HCP, 4-card heart support, balanced → 2NT over 1H
hand_j2nt = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.ACE, Suit.H),
     Card(Rank.QUEEN, Suit.H), Card(Rank.KING, Suit.D),
     Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 2, Suit.H: 4, Suit.D: 3, Suit.C: 4},
)
check(hcp(hand_j2nt) == 14, f"j2nt hand HCP = {hcp(hand_j2nt)} (expect 14)")
check(hand_shape(hand_j2nt).length(Suit.H) == 4, "j2nt hand has 4 hearts")

calls_1h_open = [make_bid(1, Suit.H), PASS]
obs = make_obs(hand_j2nt, calls=calls_1h_open, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.NT),
      f"1H - ?, 14 HCP + 4H balanced -> 2NT Jacoby (got {bid})")

# Limit raise still fires for 10-12 HCP + 4-card support (below J2NT floor)
# HAND_RESP_HEARTS from earlier: 10 HCP + 4 hearts → 3H
# (this is already covered by existing RESPONDING tests — confirm regression-free)

# Splinter: 14 HCP, 4-card spade support, singleton clubs → 4C over 1S
hand_splinter = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.S),
     Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H),
     Card(Rank.QUEEN, Suit.D)],
    {Suit.S: 4, Suit.H: 4, Suit.D: 4, Suit.C: 1},
)
check(hcp(hand_splinter) == 14, f"splinter hand HCP = {hcp(hand_splinter)} (expect 14)")
check(hand_shape(hand_splinter).length(Suit.C) == 1, "splinter hand has singleton club")

calls_1s_open = [make_bid(1, Suit.S), PASS]
obs = make_obs(hand_splinter, calls=calls_1s_open, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(4, Suit.C),
      f"1S - ?, 14 HCP + 4S + singleton C -> 4C splinter (got {bid})")

# Splinter over 1H with singleton spade → 3S (double jump)
hand_splinter_3s = _pad_hand(
    [Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H),
     Card(Rank.QUEEN, Suit.D), Card(Rank.KING, Suit.D),
     Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 1, Suit.H: 4, Suit.D: 4, Suit.C: 4},
)
check(hcp(hand_splinter_3s) == 14, f"splinter_3s HCP = {hcp(hand_splinter_3s)} (expect 14)")
obs = make_obs(hand_splinter_3s, calls=calls_1h_open, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(3, Suit.S),
      f"1H - ?, 14 HCP + 4H + singleton S -> 3S splinter (got {bid})")

# Opener answers Jacoby 2NT with shortness (singleton club) → 3C
# 16 HCP, 5H, shape 3-5-4-1 (singleton club)
# AKx S (7) + AQxxx H (6) + Kxx D (3) + x C (0) = 16
opener_j2nt_short = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
     Card(Rank.ACE, Suit.H), Card(Rank.QUEEN, Suit.H),
     Card(Rank.KING, Suit.D)],
    {Suit.S: 3, Suit.H: 5, Suit.D: 4, Suit.C: 1},
)
check(hcp(opener_j2nt_short) == 16, f"opener_j2nt_short HCP = {hcp(opener_j2nt_short)} (expect 16)")
calls_j2nt_reply = [make_bid(1, Suit.H), PASS, make_bid(2, Suit.NT), PASS]
obs = make_obs(opener_j2nt_short, calls=calls_j2nt_reply, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(3, Suit.C),
      f"1H-P-2NT-P, opener w/ singleton C -> 3C shortness (got {bid})")

# Opener answers Jacoby 2NT with min balanced → 3NT
# 13 HCP, 5H, balanced 3-5-3-2 (no shortness)
# Qxx S (2) + AKJxx H (8) + Kxx D (3) + xx C (0) = 13
opener_j2nt_min = _pad_hand(
    [Card(Rank.QUEEN, Suit.S), Card(Rank.ACE, Suit.H),
     Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H),
     Card(Rank.KING, Suit.D)],
    {Suit.S: 3, Suit.H: 5, Suit.D: 3, Suit.C: 2},
)
check(hcp(opener_j2nt_min) == 13, f"opener_j2nt_min HCP = {hcp(opener_j2nt_min)} (expect 13)")
obs = make_obs(opener_j2nt_min, calls=calls_j2nt_reply, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == make_bid(3, Suit.NT),
      f"1H-P-2NT-P, opener 13 HCP balanced -> 3NT (got {bid})")

# RKCB response encoding (direct helper test)
check(rkcb_response_for(0, False) == Suit.D, "0 keycards -> 5D")
check(rkcb_response_for(1, False) == Suit.C, "1 keycard -> 5C")
check(rkcb_response_for(2, False) == Suit.H, "2 keycards no queen -> 5H")
check(rkcb_response_for(2, True) == Suit.S, "2 keycards + queen -> 5S")
check(rkcb_response_for(3, False) == Suit.D, "3 keycards -> 5D")
check(rkcb_response_for(4, False) == Suit.C, "4 keycards -> 5C")

# RKCB decoding
r = decode_rkcb_response(make_bid(5, Suit.C))
check(r['keycards'] == (1, 4), "5C decodes to 1-or-4 keycards")
r = decode_rkcb_response(make_bid(5, Suit.H))
check(r['keycards'] == 2 and r['has_queen'] is False,
      "5H decodes to 2 keycards without queen")
r = decode_rkcb_response(make_bid(5, Suit.S))
check(r['keycards'] == 2 and r['has_queen'] is True,
      "5S decodes to 2 keycards with queen")

# Splinter table
check(splinter_short_suit(Suit.H, make_bid(4, Suit.C)) == Suit.C,
      "1H-4C is splinter in clubs")
check(splinter_short_suit(Suit.H, make_bid(3, Suit.S)) == Suit.S,
      "1H-3S is splinter in spades")
check(splinter_short_suit(Suit.S, make_bid(4, Suit.H)) == Suit.H,
      "1S-4H is splinter in hearts")
check(splinter_short_suit(Suit.S, make_bid(3, Suit.H)) is None,
      "1S-3H is NOT a splinter (single jump, not double)")
check(splinter_short_suit(Suit.H, make_bid(2, Suit.C)) is None,
      "1H-2C is natural, not splinter")

# End-to-end: RKCB response from responder. Construct auction where partner
# opened 1H, I raised to 3H, partner bid 4NT. I answer with my keycards.
# Hand: Ax S, KQxxx H, Kxx D, xxx C. Aces=1 (S), trump K=yes, queen=yes.
# Keycards = 2 + has queen = respond 5S.
hand_rkcb_2q = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.H),
     Card(Rank.QUEEN, Suit.H), Card(Rank.KING, Suit.D)],
    {Suit.S: 2, Suit.H: 5, Suit.D: 3, Suit.C: 3},
)
# Dealer=0 (partner), auction: 1H - P - 3H - P - 4NT - P - ? (me, seat 2)
calls_4nt = [
    make_bid(1, Suit.H), PASS, make_bid(3, Suit.H), PASS,
    make_bid(4, Suit.NT), PASS,
]
obs = make_obs(hand_rkcb_2q, calls=calls_4nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(5, Suit.S),
      f"4NT by partner, 2 keycards + trump queen -> 5S (got {bid})")

# Same auction shape but responder has 1 keycard → 5C
# Ax S, Qxxxx H (no K, has Q), xxx D, xxx C. 1 ace, no trump K: 1 keycard.
hand_rkcb_1k = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.H),
     Card(Rank.JACK, Suit.H)],
    {Suit.S: 2, Suit.H: 5, Suit.D: 3, Suit.C: 3},
)
obs = make_obs(hand_rkcb_1k, calls=calls_4nt, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(5, Suit.C),
      f"4NT by partner, 1 keycard -> 5C (got {bid})")


# ── COMPETITIVE DOUBLES (Batch 3) ────────────────────────────────
section("COMPETITIVE DOUBLES: NEGATIVE / SUPPORT")

# Negative double: partner opened 1H, RHO 2C overcall, I have 4+ spades
# and 8+ HCP → X.
# AQxx S (6) + xx H (0) + Kxx D (3) + Qxxx C (2) = 11 HCP, 4 spades.
hand_neg_x = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.S),
     Card(Rank.KING, Suit.D), Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 4, Suit.H: 2, Suit.D: 3, Suit.C: 4},
)
check(hcp(hand_neg_x) >= 8, f"neg_x hand HCP = {hcp(hand_neg_x)} (>=8)")
check(hand_shape(hand_neg_x).length(Suit.S) == 4, "neg_x has 4 spades")

calls_1h_2c = [make_bid(1, Suit.H), make_bid(2, Suit.C)]
obs = make_obs(hand_neg_x, calls=calls_1h_2c, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == DOUBLE,
      f"1H - (2C), 8+ HCP + 4S -> X negative (got {bid})")

# Same auction but 5-card spade → bid 2S naturally (not X)
# AQxxx S + xx H + Kxx D + Qxx C = 10 HCP, 5 spades
hand_5s = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.S),
     Card(Rank.JACK, Suit.S), Card(Rank.KING, Suit.D),
     Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 5, Suit.H: 2, Suit.D: 3, Suit.C: 3},
)
check(hand_shape(hand_5s).length(Suit.S) == 5, "5s hand has 5 spades")
obs = make_obs(hand_5s, calls=calls_1h_2c, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == make_bid(2, Suit.S),
      f"1H - (2C), 10 HCP + 5S -> 2S natural, not X (got {bid})")

# Negative double over 1C - (1D) showing 4+ hearts, 6+ HCP
# Kxxx H (3) + Qxxx S (2) + xx D (0) + Jxx C (1) = 6 HCP, 4 hearts
hand_neg_x_h = _pad_hand(
    [Card(Rank.KING, Suit.H), Card(Rank.QUEEN, Suit.S),
     Card(Rank.JACK, Suit.C)],
    {Suit.S: 4, Suit.H: 4, Suit.D: 2, Suit.C: 3},
)
check(hcp(hand_neg_x_h) >= 6, f"neg_x_h HCP = {hcp(hand_neg_x_h)} (>=6)")

calls_1c_1d = [make_bid(1, Suit.C), make_bid(1, Suit.D)]
obs = make_obs(hand_neg_x_h, calls=calls_1c_1d, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == DOUBLE,
      f"1C - (1D), 6+ HCP + 4H -> X negative (got {bid})")

# Weak hand: 3 HCP → no neg X, PASS
hand_weak_neg = _pad_hand(
    [Card(Rank.JACK, Suit.S)],
    {Suit.S: 4, Suit.H: 4, Suit.D: 3, Suit.C: 2},
)
check(hcp(hand_weak_neg) <= 5, f"weak_neg HCP = {hcp(hand_weak_neg)}")
obs = make_obs(hand_weak_neg, calls=calls_1h_2c, dealer=0, player=2)
bid = resp.bid(obs)
check(bid == PASS,
      f"1H - (2C), 1 HCP + 4S -> PASS (too weak) (got {bid})")

# Opener's response to partner's negative double.
# Me = N (seat 0). Opened 1C. RHO (E) 1D. Partner (S) X. LHO (W) P. My turn.
# I have 3-card heart support + min hand → 2H.
# AKxxx C + Qxx H + Axx S + xx D. Wait I opened 1C so I need 3+ clubs.
# Let me redo: AKx S + Qxx H + xx D + AKxxx C = A+K+Q+A+K = 17 HCP.
# Simpler for test: 13 HCP 1C opening with 3 hearts.
# Axx S + Qxx H + Kx D + AQxxx C = 4+2+3+6 = 15 HCP, 3S/3H/2D/5C.
# Need exactly 3 hearts with rebid values. Let me try:
# Kxx S + Qxx H + Axx D + AJxx C = 3+2+4+5 = 14 HCP. Balanced-ish 3-3-3-4.
hand_opener_neg_resp = _pad_hand(
    [Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.H),
     Card(Rank.ACE, Suit.D), Card(Rank.ACE, Suit.C),
     Card(Rank.JACK, Suit.C)],
    {Suit.S: 3, Suit.H: 3, Suit.D: 3, Suit.C: 4},
)
check(hand_shape(hand_opener_neg_resp).length(Suit.H) == 3,
      "opener_neg_resp has 3 hearts")

# Auction: N(me)=1C, E=1D, S(partner)=X (negative), W=P, now me.
calls_neg_response = [
    make_bid(1, Suit.C), make_bid(1, Suit.D), DOUBLE, PASS,
]
obs = make_obs(hand_opener_neg_resp, calls=calls_neg_response,
               dealer=0, player=0)
bid = opener.bid(obs)
# Partner promised 4+ hearts. I have 3. Bid hearts at some level.
check(bid.strain == Suit.H and bid.level >= 2 and bid.level <= 3
      and not bid.special,
      f"1C-(1D)-X-(P), 14 HCP + 3H -> raise hearts (got {bid})")

# Support double: I opened 1C, partner responded 1H, opp overcalled 1S.
# With exactly 3 hearts, I should double.
# Axx S + Qxx H + Kxxx D + AKx C = 4+2+3+7 = 16 HCP, 3-3-4-3.
hand_support_x = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.H),
     Card(Rank.KING, Suit.D), Card(Rank.ACE, Suit.C),
     Card(Rank.KING, Suit.C)],
    {Suit.S: 3, Suit.H: 3, Suit.D: 4, Suit.C: 3},
)
check(hand_shape(hand_support_x).length(Suit.H) == 3,
      "support_x has exactly 3 hearts")

# N(me) dealer. Auction: N=1C, E=P, S(partner)=1H, W=1S, now me.
calls_sup_x = [
    make_bid(1, Suit.C), PASS, make_bid(1, Suit.H), make_bid(1, Suit.S),
]
obs = make_obs(hand_support_x, calls=calls_sup_x, dealer=0, player=0)
bid = opener.bid(obs)
check(bid == DOUBLE,
      f"1C-(P)-1H-(1S), 16 HCP + exactly 3H -> X support (got {bid})")

# Same scenario but 4 hearts — should raise 2H, NOT support X.
# Axx S + Qxxx H + Kxx D + AKx C = 4+2+3+7 = 16, 3-4-3-3
hand_4h_raise = _pad_hand(
    [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.H),
     Card(Rank.KING, Suit.D), Card(Rank.ACE, Suit.C),
     Card(Rank.KING, Suit.C)],
    {Suit.S: 3, Suit.H: 4, Suit.D: 3, Suit.C: 3},
)
check(hand_shape(hand_4h_raise).length(Suit.H) == 4,
      "4h_raise has 4 hearts")
obs = make_obs(hand_4h_raise, calls=calls_sup_x, dealer=0, player=0)
bid = opener.bid(obs)
check(bid != DOUBLE,
      f"1C-(P)-1H-(1S), 16 HCP + 4H -> NOT X (got {bid})")

# Responder answering partner's support double.
# Partner N=1C, E=P, me(S)=1H, W=1S, N=X (support), E=P, me to bid.
# I have 5 hearts + 10 HCP → raise to 3H (fit known = 8+).
# x S + AJxxx H + Kxx D + Qxxx C = 1+5+3+2 = 11 HCP, 1-5-3-4
hand_resp_sup_x = _pad_hand(
    [Card(Rank.ACE, Suit.H), Card(Rank.JACK, Suit.H),
     Card(Rank.KING, Suit.D), Card(Rank.QUEEN, Suit.C)],
    {Suit.S: 1, Suit.H: 5, Suit.D: 4, Suit.C: 3},
)
check(hand_shape(hand_resp_sup_x).length(Suit.H) == 5,
      "resp_sup_x has 5 hearts")

calls_resp_sup_x = [
    make_bid(1, Suit.C), PASS, make_bid(1, Suit.H), make_bid(1, Suit.S),
    DOUBLE, PASS,
]
obs = make_obs(hand_resp_sup_x, calls=calls_resp_sup_x,
               dealer=0, player=2)
bid = resp.bid(obs)
check(bid.strain == Suit.H and bid.level >= 3 and not bid.special,
      f"1C-(P)-1H-(1S)-X-(P), 5H + ~11 HCP -> raise hearts (got {bid})")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
