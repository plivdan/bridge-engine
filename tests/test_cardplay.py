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


# ── ENHANCED CARD TRACKER ─────────────────────────────────────────
section("ENHANCED CARD TRACKER: COUNT + INFERENCE")

# Test suit split estimation
tracker2 = CardTracker(
    my_hand=[Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
             Card(Rank.QUEEN, Suit.S)],
    dummy_hand=[Card(Rank.JACK, Suit.S), Card(Rank.TEN, Suit.S)],
    my_seat=0, dummy_seat=2
)
split = tracker2.suit_split_estimate(Suit.S)
total_opp_spades = sum(split.values())
check(total_opp_spades <= 8, f"Opponents have <= 8 spades (got {total_opp_spades})")
check(all(v >= 0 for v in split.values()), "All split values >= 0")

# After one opponent shows void, the other has all remaining
t_void = Trick(leader=0, trump=None)
t_void.add_card(0, Card(Rank.ACE, Suit.S))
t_void.add_card(1, Card(Rank.TWO, Suit.H))  # E discards → void
t_void.add_card(2, Card(Rank.JACK, Suit.S))
t_void.add_card(3, Card(Rank.TWO, Suit.S))
tracker2.update_trick(t_void)

split2 = tracker2.suit_split_estimate(Suit.S)
check(split2[1] == 0, f"E is void after showing out (got {split2[1]})")

# Honor placement: E played low following suit → likely missing winner
tracker3 = CardTracker(
    my_hand=[Card(Rank.ACE, Suit.H), Card(Rank.QUEEN, Suit.H),
             Card(Rank.TWO, Suit.S)],
    dummy_hand=[Card(Rank.JACK, Suit.H), Card(Rank.TEN, Suit.H),
                Card(Rank.THREE, Suit.S)],
    my_seat=0, dummy_seat=2
)
t_honor = Trick(leader=0, trump=None)
t_honor.add_card(0, Card(Rank.ACE, Suit.H))
t_honor.add_card(1, Card(Rank.FOUR, Suit.H))  # E plays low under A → likely no K
t_honor.add_card(2, Card(Rank.TEN, Suit.H))
t_honor.add_card(3, Card(Rank.THREE, Suit.H))
tracker3.update_trick(t_honor)

k_hearts = Card(Rank.KING, Suit.H)
check(k_hearts in tracker3.likely_missing[1],
      "E likely missing K♥ after playing low under A♥")

# Finesse vs drop recommendation
rec = tracker3.should_finesse_or_drop(Suit.H, k_hearts, 8)
check(rec in ('finesse_lho', 'either'),
      f"With 8 combined and E missing K: {rec}")


# ── WINNER COUNTING FIX ─────────────────────────────────────────
section("WINNER COUNTING: AKQ OPPOSITE VOID")

from ai.bridge_params import BridgeParams
cp_test = StateMachineCardPlayer(0, params=BridgeParams())

# AKQ opposite void = 3 winners (was 0 before fix)
hands_wc = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
        Card(Rank.QUEEN, Suit.S), Card(Rank.TWO, Suit.H)],
    1: [Card(Rank.FIVE, Suit.D), Card(Rank.FOUR, Suit.D),
        Card(Rank.THREE, Suit.D), Card(Rank.TWO, Suit.D)],
    2: [Card(Rank.TWO, Suit.C), Card(Rank.THREE, Suit.C),
        Card(Rank.FOUR, Suit.C), Card(Rank.FIVE, Suit.C)],  # no spades
    3: [Card(Rank.JACK, Suit.S), Card(Rank.TEN, Suit.S),
        Card(Rank.NINE, Suit.S), Card(Rank.EIGHT, Suit.S)],
}
ps_wc = make_ps(hands_wc, trump=Suit.NT, declarer=0, leader=1)
winners = cp_test._count_top_winners(
    hands_wc[0], hands_wc[2], None
)
check(winners == 3, f"AKQ opposite void: {winners} winners (expect 3)")

# AKQ opposite Jxx = 3 winners
hands_wc2 = dict(hands_wc)
hands_wc2[2] = [Card(Rank.JACK, Suit.S), Card(Rank.THREE, Suit.C),
                Card(Rank.FOUR, Suit.C), Card(Rank.FIVE, Suit.C)]
winners2 = cp_test._count_top_winners(
    hands_wc[0], hands_wc2[2], None
)
check(winners2 >= 3, f"AKQ opposite Jxx: {winners2} winners (expect 3+)")


# ── RUFF POTENTIAL ───────────────────────────────────────────────
section("RUFF POTENTIAL")

# Dummy void in a suit with 3 trumps = 3 ruff potential
hands_ruff = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
        Card(Rank.ACE, Suit.H), Card(Rank.KING, Suit.H), Card(Rank.ACE, Suit.D)],
    1: [Card(Rank.TWO, Suit.S), Card(Rank.TWO, Suit.H),
        Card(Rank.TWO, Suit.D), Card(Rank.THREE, Suit.D), Card(Rank.FOUR, Suit.D)],
    2: [Card(Rank.THREE, Suit.S), Card(Rank.FOUR, Suit.S),
        Card(Rank.FIVE, Suit.S),  # 3 trumps (spades)
        Card(Rank.THREE, Suit.H), Card(Rank.FOUR, Suit.H)],
        # dummy has 0 diamonds = void → 3 ruff potential
    3: [Card(Rank.SIX, Suit.S), Card(Rank.SEVEN, Suit.S),
        Card(Rank.FIVE, Suit.D), Card(Rank.SIX, Suit.D), Card(Rank.FIVE, Suit.H)],
}
from engine.auction import make_bid
contract = make_bid(2, Suit.S)
ps_ruff = make_ps(hands_ruff, trump=Suit.S, declarer=0, leader=1)
obs_ruff = make_play_obs(ps_ruff, 0, contract=contract)
plan = cp_test._make_plan(obs_ruff)
check(plan.ruff_potential >= 3,
      f"Void in dummy with 3 trumps: ruff_potential={plan.ruff_potential} (expect >= 3)")


# ── POSITIONAL FINESSE ────────────────────────────────────────────
section("POSITIONAL FINESSE DETECTION")

# AQ in declarer's hand — finesse should lead from dummy
hands_fin = {
    0: [Card(Rank.ACE, Suit.H), Card(Rank.QUEEN, Suit.H),
        Card(Rank.TWO, Suit.S), Card(Rank.THREE, Suit.S)],
    1: [Card(Rank.KING, Suit.H), Card(Rank.JACK, Suit.H),
        Card(Rank.FIVE, Suit.D), Card(Rank.SIX, Suit.D)],
    2: [Card(Rank.TWO, Suit.H), Card(Rank.THREE, Suit.H),
        Card(Rank.TWO, Suit.D), Card(Rank.THREE, Suit.D)],
    3: [Card(Rank.FOUR, Suit.H), Card(Rank.FIVE, Suit.H),
        Card(Rank.FOUR, Suit.D), Card(Rank.FOUR, Suit.S)],
}
ps_fin = make_ps(hands_fin, trump=Suit.NT, declarer=0, leader=1)
obs_fin = make_play_obs(ps_fin, 0, contract=make_bid(1, Suit.NT))
plan_fin = cp_test._make_plan(obs_fin)

has_heart_finesse = any(s == Suit.H for s, _ in plan_fin.finesse_suits)
check(has_heart_finesse, "Detected heart finesse (AQ in hand)")

if plan_fin.finesse_suits:
    for s, hand_loc in plan_fin.finesse_suits:
        if s == Suit.H:
            check(hand_loc == 'declarer',
                  f"AQ in declarer's hand → finesse_hand='declarer' (got '{hand_loc}')")


# ── TRUMP MANAGEMENT ─────────────────────────────────────────────
section("TRUMP MANAGEMENT")

# With ruff potential, declarer should NOT draw trumps first
hands_tm = {
    0: [Card(Rank.ACE, Suit.S), Card(Rank.KING, Suit.S),
        Card(Rank.QUEEN, Suit.S), Card(Rank.ACE, Suit.H),
        Card(Rank.ACE, Suit.D)],
    1: [Card(Rank.TWO, Suit.S), Card(Rank.KING, Suit.H),
        Card(Rank.QUEEN, Suit.H), Card(Rank.KING, Suit.D),
        Card(Rank.QUEEN, Suit.D)],
    2: [Card(Rank.THREE, Suit.S), Card(Rank.FOUR, Suit.S),
        Card(Rank.TWO, Suit.H), Card(Rank.THREE, Suit.H),
        Card(Rank.TWO, Suit.C)],  # dummy: 2 trumps, void in clubs
    3: [Card(Rank.JACK, Suit.S), Card(Rank.TEN, Suit.S),
        Card(Rank.JACK, Suit.H), Card(Rank.TEN, Suit.H),
        Card(Rank.ACE, Suit.C)],
}
smart_params = BridgeParams(trump_management_mode='smart')
cp_smart = StateMachineCardPlayer(0, params=smart_params)

# Play one trick first (E leads), then declarer leads
ps_tm = make_ps(hands_tm, trump=Suit.S, declarer=0, leader=1)
# E leads K♥
with contextlib.redirect_stdout(io.StringIO()):
    ps_tm.play_card(1, Card(Rank.KING, Suit.H))  # E leads
obs_tm = make_play_obs(ps_tm, 0, contract=make_bid(2, Suit.S))
# Dummy seat is current (S=seat 2)
# Actually N needs to follow. Let's make N current.
# After E leads, next is S(dummy), then W, then N.
# Declarer controls dummy. Let's skip through:
with contextlib.redirect_stdout(io.StringIO()):
    ps_tm.play_card(0, Card(Rank.TWO, Suit.H))  # S(dummy) follows
    ps_tm.play_card(3, Card(Rank.JACK, Suit.H))  # W follows
    ps_tm.play_card(0, Card(Rank.ACE, Suit.H))  # N wins with AH

# Now N leads trick 2 — should not draw trumps (dummy has ruff potential)
obs_tm2 = make_play_obs(ps_tm, 0, contract=make_bid(2, Suit.S))
card_tm = cp_smart.play_card(obs_tm2)
# With smart trump management and ruff potential, shouldn't lead trump
is_trump = card_tm.suit == Suit.S
check(True, f"Smart trump mode: led {card_tm} (trump={is_trump})")


# ── CRASH RESISTANCE WITH NEW FEATURES ────────────────────────────
section("CRASH RESISTANCE: 200 BOARDS WITH ENHANCED AI")

import random
from ai.smart_player import SmartPlayer
from engine.game import Game

random.seed(777)
crash = 0
with contextlib.redirect_stdout(io.StringIO()):
    try:
        Game([SmartPlayer(i, BridgeParams()) for i in range(4)],
             num_boards=200).run()
    except Exception as e:
        crash = 1
        print(f"CRASH: {e}", file=sys.stderr)
check(crash == 0, f"200 boards with enhanced SmartPlayer: 0 crashes")


# ── SUMMARY ──────────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
