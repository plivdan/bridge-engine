"""Card play agent correctness tests.

Tests known bridge positions with expected correct plays.
Uses the same check/section framework as test_bridge.py.
"""

import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.card import Card, Suit, Rank
from engine.play import PlayState, Trick
from engine.state import GameState
from engine.auction import AuctionState, make_bid, PASS
from ai.cardplay_agent import StateMachineCardPlayer, CardTracker
from ai.hand_eval import hcp

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

def make_play_obs(ps, player, calls=None, contract=None):
    """Build a minimal observation dict from a PlayState."""
    dummy_hand = list(ps.hands[ps.dummy])
    seat = ps.current_seat
    return {
        'hand': list(ps.hands[player]),
        'valid_cards': ps.valid_cards(seat) if seat is not None else [],
        'dummy_hand': dummy_hand,
        'tricks_ns': ps.tricks_ns,
        'tricks_ew': ps.tricks_ew,
        'completed_tricks': list(ps.tricks),
        'current_trick': ps.current_trick,
        'trump': ps.trump,
        'current_player': ps.current_player,
        'current_seat': ps.current_seat,
        'declarer': ps.declarer,
        'calls': calls or [],
        'dealer': 0,
        'player': player,
        'contract': contract,
        'vulnerable': {'NS': False, 'EW': False},
    }


def make_ps(hands, trump=Suit.NT, declarer=0, leader=1):
    """Create a PlayState with preset hands, suppressing output."""
    dummy = (declarer + 2) % 4
    with contextlib.redirect_stdout(io.StringIO()):
        return PlayState(hands=hands, trump=trump, declarer=declarer,
                         dummy=dummy, leader=leader)


# ── OPENING LEAD: TOP OF SEQUENCE ────────────────────────────────
section("OPENING LEAD: TOP OF SEQUENCE")

# East (seat 1) has KQJxx in spades, should lead K
hands_seq = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.TWO, Suit.H),
        Card(Rank.THREE, Suit.D), Card(Rank.FOUR, Suit.C)],
    1: [Card(Rank.KING, Suit.S), Card(Rank.QUEEN, Suit.S),
        Card(Rank.JACK, Suit.S), Card(Rank.NINE, Suit.S)],
    2: [Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.H),
        Card(Rank.SIX, Suit.D), Card(Rank.SEVEN, Suit.C)],
    3: [Card(Rank.EIGHT, Suit.S), Card(Rank.ACE, Suit.H),
        Card(Rank.ACE, Suit.D), Card(Rank.ACE, Suit.C)],
}

ps_seq = make_ps(hands_seq, trump=Suit.NT, declarer=0, leader=1)
cp = StateMachineCardPlayer(1)
obs = make_play_obs(ps_seq, 1)
card = cp.play_card(obs)
check(card == Card(Rank.KING, Suit.S),
      f"KQJ sequence: lead K (got {card})")


# ── OPENING LEAD: 4TH BEST vs NT ─────────────────────────────────
section("OPENING LEAD: 4TH BEST vs NT")

# East has A K 8 7 5 in hearts, should lead 7 (4th best)
hands_4th = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.TWO, Suit.S),
        Card(Rank.THREE, Suit.D), Card(Rank.FOUR, Suit.C), Card(Rank.SIX, Suit.C)],
    1: [Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H),
        Card(Rank.EIGHT, Suit.H), Card(Rank.SEVEN, Suit.H), Card(Rank.FIVE, Suit.H)],
    2: [Card(Rank.TEN, Suit.S), Card(Rank.FIVE, Suit.S),
        Card(Rank.SIX, Suit.D), Card(Rank.SEVEN, Suit.D), Card(Rank.TWO, Suit.C)],
    3: [Card(Rank.EIGHT, Suit.S), Card(Rank.QUEEN, Suit.H),
        Card(Rank.ACE, Suit.D), Card(Rank.NINE, Suit.C), Card(Rank.THREE, Suit.C)],
}

ps_4th = make_ps(hands_4th, trump=Suit.NT, declarer=0, leader=1)
cp2 = StateMachineCardPlayer(1)
obs = make_play_obs(ps_4th, 1)
card = cp2.play_card(obs)
# 4th best from AK875 is 7
check(card.suit == Suit.H, f"4th best: lead hearts (got {card})")
check(card == Card(Rank.SEVEN, Suit.H),
      f"4th best from AK875: lead 7 (got {card})")


# ── THIRD-HAND HIGH ──────────────────────────────────────────────
section("THIRD-HAND HIGH")

# West (seat 3) leads 3♠. Declarer (N) plays TEN, beating partner.
# East (seat 1) should play K to retake the trick.
# Order: W(3) → N(0) → E(1) → S(2/dummy)
hands_3rd = {
    0: [Card(Rank.TEN, Suit.S), Card(Rank.FOUR, Suit.S),
        Card(Rank.FIVE, Suit.H), Card(Rank.SIX, Suit.D)],  # declarer
    1: [Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
        Card(Rank.EIGHT, Suit.H), Card(Rank.SEVEN, Suit.D)],  # east (us)
    2: [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.S),
        Card(Rank.TWO, Suit.H), Card(Rank.THREE, Suit.D)],  # dummy
    3: [Card(Rank.NINE, Suit.S), Card(Rank.THREE, Suit.S),
        Card(Rank.ACE, Suit.H), Card(Rank.ACE, Suit.D)],  # west (partner)
}

ps_3rd = make_ps(hands_3rd, trump=Suit.NT, declarer=0, leader=3)
# West leads 3♠, then declarer plays T♠ (beats partner's 3)
with contextlib.redirect_stdout(io.StringIO()):
    ps_3rd.play_card(3, Card(Rank.THREE, Suit.S))
    ps_3rd.play_card(0, Card(Rank.TEN, Suit.S))

# East should play K to win the trick (partner led, partner not winning)
cp3 = StateMachineCardPlayer(1)
obs = make_play_obs(ps_3rd, 1)
card = cp3.play_card(obs)
check(card == Card(Rank.KING, Suit.S),
      f"third-hand high: play K to beat T (got {card})")


# ── SECOND-HAND LOW ──────────────────────────────────────────────
section("SECOND-HAND LOW")

# Declarer (North, seat 0) leads small spade.
# East (seat 1) is second hand and should play low.
# Order: N(0) → E(1) → S(2/dummy) → W(3)
hands_2nd = {
    0: [Card(Rank.FIVE, Suit.S), Card(Rank.THREE, Suit.S),
        Card(Rank.ACE, Suit.H), Card(Rank.ACE, Suit.D)],  # declarer
    1: [Card(Rank.KING, Suit.S), Card(Rank.JACK, Suit.S),
        Card(Rank.SEVEN, Suit.H), Card(Rank.SIX, Suit.D)],  # east (us, second hand)
    2: [Card(Rank.ACE, Suit.S), Card(Rank.QUEEN, Suit.S),
        Card(Rank.FOUR, Suit.H), Card(Rank.THREE, Suit.D)],  # dummy
    3: [Card(Rank.NINE, Suit.S), Card(Rank.EIGHT, Suit.S),
        Card(Rank.TEN, Suit.H), Card(Rank.TEN, Suit.D)],  # west
}

ps_2nd = make_ps(hands_2nd, trump=Suit.NT, declarer=0, leader=0)
# Declarer leads 3♠
with contextlib.redirect_stdout(io.StringIO()):
    ps_2nd.play_card(0, Card(Rank.THREE, Suit.S))

# East (seat 1) second hand should play low (J not K)
cp4 = StateMachineCardPlayer(1)
obs = make_play_obs(ps_2nd, 1)
card = cp4.play_card(obs)
check(card == Card(Rank.JACK, Suit.S),
      f"second-hand low: play J not K (got {card})")


# ── COVER AN HONOR ───────────────────────────────────────────────
section("COVER AN HONOR")

# Declarer leads Q♠. East (seat 1, second hand) has K♠ and should cover.
# Order: N(0) → E(1) → S(2/dummy) → W(3)
hands_cover = {
    0: [Card(Rank.QUEEN, Suit.S), Card(Rank.JACK, Suit.S),
        Card(Rank.ACE, Suit.H), Card(Rank.ACE, Suit.D)],  # declarer
    1: [Card(Rank.KING, Suit.S), Card(Rank.FIVE, Suit.S),
        Card(Rank.SEVEN, Suit.H), Card(Rank.SIX, Suit.D)],  # east (us)
    2: [Card(Rank.ACE, Suit.S), Card(Rank.TEN, Suit.S),
        Card(Rank.FOUR, Suit.H), Card(Rank.THREE, Suit.D)],  # dummy
    3: [Card(Rank.NINE, Suit.S), Card(Rank.EIGHT, Suit.S),
        Card(Rank.TEN, Suit.H), Card(Rank.TEN, Suit.D)],  # west
}

ps_cover = make_ps(hands_cover, trump=Suit.NT, declarer=0, leader=0)
with contextlib.redirect_stdout(io.StringIO()):
    ps_cover.play_card(0, Card(Rank.QUEEN, Suit.S))

cp5 = StateMachineCardPlayer(1)
obs = make_play_obs(ps_cover, 1)
card = cp5.play_card(obs)
check(card == Card(Rank.KING, Suit.S),
      f"cover honor: play K over Q (got {card})")


# ── DECLARER: WIN CHEAPLY ────────────────────────────────────────
section("DECLARER: WIN CHEAPLY")

# Declarer follows to opponent's lead and should win with cheapest card
hands_cheap = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
        Card(Rank.QUEEN, Suit.S), Card(Rank.ACE, Suit.H)],  # declarer
    1: [Card(Rank.FIVE, Suit.S), Card(Rank.THREE, Suit.S),
        Card(Rank.SEVEN, Suit.H), Card(Rank.SIX, Suit.D)],  # east
    2: [Card(Rank.TEN, Suit.S), Card(Rank.FOUR, Suit.S),
        Card(Rank.FOUR, Suit.H), Card(Rank.THREE, Suit.D)],  # dummy
    3: [Card(Rank.JACK, Suit.S), Card(Rank.NINE, Suit.S),
        Card(Rank.TEN, Suit.H), Card(Rank.TEN, Suit.D)],  # west
}

ps_cheap = make_ps(hands_cheap, trump=Suit.NT, declarer=0, leader=3)
# West leads J♠
with contextlib.redirect_stdout(io.StringIO()):
    ps_cheap.play_card(3, Card(Rank.JACK, Suit.S))

# Declarer (N, seat 0) should win cheaply with Q, not A or K
cp6 = StateMachineCardPlayer(0)
obs = make_play_obs(ps_cheap, 0)
card = cp6.play_card(obs)
check(card == Card(Rank.QUEEN, Suit.S),
      f"win cheaply: play Q over J (got {card})")


# ── DEFENDER: RUFF WHEN VOID ─────────────────────────────────────
section("DEFENDER: RUFF WHEN VOID")

# West (seat 3) leads spade. N plays. East (seat 1) is void in spades,
# has trumps (hearts). Should ruff.
# Order: W(3) → N(0) → E(1) → S(2/dummy)
hands_ruff = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
        Card(Rank.FIVE, Suit.H), Card(Rank.SIX, Suit.D)],  # declarer
    1: [Card(Rank.TWO, Suit.H), Card(Rank.THREE, Suit.H),
        Card(Rank.SEVEN, Suit.H), Card(Rank.SIX, Suit.H)],  # east (void S, has H)
    2: [Card(Rank.TEN, Suit.S), Card(Rank.FOUR, Suit.S),
        Card(Rank.EIGHT, Suit.H), Card(Rank.THREE, Suit.D)],  # dummy
    3: [Card(Rank.NINE, Suit.S), Card(Rank.THREE, Suit.S),
        Card(Rank.ACE, Suit.H), Card(Rank.ACE, Suit.D)],  # west
}

ps_ruff = make_ps(hands_ruff, trump=Suit.H, declarer=0, leader=3)
# W leads 9♠, then declarer (N) plays A♠
with contextlib.redirect_stdout(io.StringIO()):
    ps_ruff.play_card(3, Card(Rank.NINE, Suit.S))
    ps_ruff.play_card(0, Card(Rank.ACE, Suit.S))

# East should ruff with lowest trump
cp7 = StateMachineCardPlayer(1)
obs = make_play_obs(ps_ruff, 1)
card = cp7.play_card(obs)
check(card.suit == Suit.H,
      f"ruff when void: play a trump (got {card})")
check(card == Card(Rank.TWO, Suit.H),
      f"ruff with lowest trump: 2H (got {card})")


# ── CARD TRACKER ─────────────────────────────────────────────────
section("CARD TRACKER")

from engine.card import DECK

my_hand = [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S)]
dummy_hand = [Card(Rank.QUEEN, Suit.S), Card(Rank.JACK, Suit.S)]
tracker = CardTracker(my_hand, dummy_hand, my_seat=0, dummy_seat=2)

check(len(tracker.unknown) == 48,
      f"tracker starts with 48 unknown cards (got {len(tracker.unknown)})")

# Simulate a trick
t = Trick(leader=1, trump=None)
t.add_card(1, Card(Rank.TEN, Suit.S))
t.add_card(2, Card(Rank.JACK, Suit.S))
t.add_card(3, Card(Rank.NINE, Suit.S))
t.add_card(0, Card(Rank.ACE, Suit.S))
tracker.update_trick(t)

check(Card(Rank.TEN, Suit.S) in tracker.played,
      "10S marked as played after trick")
check(len(tracker.played) == 4,
      f"4 cards played after 1 trick (got {len(tracker.played)})")

# Simulate void detection
t2 = Trick(leader=0, trump=None)
t2.add_card(0, Card(Rank.KING, Suit.S))
t2.add_card(1, Card(Rank.TWO, Suit.H))  # East discards heart → void in spades
t2.add_card(2, Card(Rank.QUEEN, Suit.S))
t2.add_card(3, Card(Rank.EIGHT, Suit.S))
tracker.update_trick(t2)

check(tracker.opponent_is_void(1, Suit.S),
      "East shown void in spades after discarding")
check(not tracker.opponent_is_void(3, Suit.S),
      "West not void in spades (followed suit)")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
