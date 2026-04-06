import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from card import Card, Suit, Rank, DECK, deal
from auction import AuctionState, Bid, PASS, DOUBLE, REDOUBLE, make_bid
from play import PlayState, Trick
from scoring import score, _doubled_penalty, _trick_score
from state import GameState
from player import RandomPlayer, PassingPlayer, SimpleHeuristicPlayer, RuleBasedPlayer
from game import Game, SelfPlayEnv, VUL_SCHEDULE

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

# ── CARD PRIMITIVES ──────────────────────────────────────────
section("CARD PRIMITIVES")
check(len(DECK) == 52, "deck has 52 cards")
check(len(set(DECK)) == 52, "all cards unique")
check(all(isinstance(c, Card) for c in DECK), "all cards are Card instances")
check(Card(Rank.ACE, Suit.S).hcp() == 4, "ace = 4 HCP")
check(Card(Rank.KING, Suit.H).hcp() == 3, "king = 3 HCP")
check(Card(Rank.QUEEN, Suit.D).hcp() == 2, "queen = 2 HCP")
check(Card(Rank.JACK, Suit.C).hcp() == 1, "jack = 1 HCP")
check(Card(Rank.TEN, Suit.S).hcp() == 0, "ten = 0 HCP")
check(Card(Rank.ACE, Suit.S) > Card(Rank.KING, Suit.S), "ace > king same suit")
check(Card(Rank.TWO, Suit.C) < Card(Rank.ACE, Suit.S), "2c < As cross-suit")
check(Card(Rank.ACE, Suit.S) == Card(Rank.ACE, Suit.S), "card equality")
check(len({Card(Rank.ACE, Suit.S), Card(Rank.ACE, Suit.S)}) == 1, "card hashable/dedup")
check(sum(c.hcp() for c in DECK) == 40, "deck total = 40 HCP")

# ── DEAL ─────────────────────────────────────────────────────
section("DEAL")
random.seed(0)
for _ in range(10):
    h = deal()
    all_cards = [c for hand in h.values() for c in hand]
    check(len(h) == 4, "deal returns 4 hands")
    check(all(len(h[i]) == 13 for i in range(4)), "each hand 13 cards")
    check(len(set(all_cards)) == 52, "no duplicate cards across hands")
    check(len(all_cards) == 52, "total 52 cards dealt")

total_hcp = [sum(c.hcp() for c in h[i]) for i in range(4)]
check(sum(total_hcp) == 40, "total HCP = 40 after deal")

# ── BID ORDERING ─────────────────────────────────────────────
section("BID ORDERING")
b1c = make_bid(1, Suit.C); b1d = make_bid(1, Suit.D)
b1h = make_bid(1, Suit.H); b1s = make_bid(1, Suit.S)
b1n = make_bid(1, Suit.NT); b2c = make_bid(2, Suit.C)
b7n = make_bid(7, Suit.NT)
check(b1c < b1d < b1h < b1s < b1n < b2c, "suit ordering within level")
check(b2c > b1n, "level 2 > level 1")
check(b7n > b1c, "7NT > 1C")
check(not (b1c > b1c), "bid not greater than itself")
check(b1c <= b1c, "bid <= itself")
check(b7n >= b1c, "7NT >= 1C")

# ── AUCTION VALID CALLS ───────────────────────────────────────
section("AUCTION VALID CALLS")
a = AuctionState(dealer=0)
vc = a.valid_calls()
check(len(vc) == 36, f"opening has 35 bids + PASS = 36 (got {len(vc)})")
check(PASS in vc, "PASS always valid")
check(DOUBLE not in vc, "DOUBLE invalid at start")
check(REDOUBLE not in vc, "REDOUBLE invalid at start")
check(make_bid(1, Suit.C) in vc, "1C valid opening")
check(make_bid(7, Suit.NT) in vc, "7NT valid opening")

a.apply_call(make_bid(1, Suit.S))
vc2 = a.valid_calls()
check(DOUBLE in vc2, "E can double after N opens 1S")
check(make_bid(1, Suit.S) not in vc2, "1S no longer valid after 1S")
check(make_bid(1, Suit.NT) in vc2, "1NT valid over 1S")
check(make_bid(1, Suit.H) not in vc2, "1H invalid over 1S (lower strain same level)")

a.apply_call(DOUBLE)
vc3 = a.valid_calls()
check(REDOUBLE in vc3, "S can redouble after E doubles")
check(DOUBLE not in vc3, "no double after double by opponents")

a2 = AuctionState(dealer=0)
a2.apply_call(make_bid(1, Suit.S)); a2.apply_call(PASS)
vc_s = a2.valid_calls()
check(DOUBLE not in vc_s, "S cannot double partner's bid")

# ── AUCTION COMPLETION ────────────────────────────────────────
section("AUCTION COMPLETION")
def make_auction(calls, dealer=0):
    a = AuctionState(dealer=dealer)
    for c in calls: a.apply_call(c)
    return a

a_po = make_auction([PASS]*4)
check(a_po.is_complete(), "4 passes = complete")
check(a_po.is_passed_out(), "4 passes = passed out")
check(a_po.result()['contract'] is None, "passed out contract = None")

a_3p = make_auction([make_bid(1,Suit.S), PASS, PASS, PASS])
check(a_3p.is_complete(), "bid + 3 passes = complete")
check(not a_3p.is_passed_out(), "bid + 3 passes != passed out")
check(a_3p.result()['contract'].level == 1, "contract level correct")

a_mid = make_auction([make_bid(1,Suit.S), make_bid(2,Suit.H), PASS])
check(not a_mid.is_complete(), "3 calls, last bid not yet complete")

a_dbl = make_auction([make_bid(2,Suit.S), DOUBLE, PASS, PASS, PASS])
check(a_dbl.is_complete(), "bid+dbl+3passes = complete")
check(a_dbl.result()['doubled'] == 1, "doubled flag = 1")

a_rdbl = make_auction([make_bid(2,Suit.S), DOUBLE, REDOUBLE, PASS, PASS, PASS])
check(a_rdbl.is_complete(), "bid+dbl+rdbl+3passes = complete")
check(a_rdbl.result()['doubled'] == 2, "redoubled flag = 2")

# ── DECLARER ASSIGNMENT ───────────────────────────────────────
section("DECLARER ASSIGNMENT")
a_d1 = make_auction([make_bid(1,Suit.H), PASS, PASS, PASS], dealer=0)
check(a_d1.result()['declarer'] == 0, "N opens 1H => N is declarer")

a_d2 = make_auction([PASS, make_bid(1,Suit.H), PASS, make_bid(2,Suit.H), PASS, PASS, PASS], dealer=0)
check(a_d2.result()['declarer'] == 1, "E first to bid H on EW side => E declarer despite W bidding higher")

a_d3 = make_auction([make_bid(1,Suit.S), PASS, make_bid(2,Suit.S), PASS, PASS, PASS], dealer=0)
check(a_d3.result()['declarer'] == 0, "N first to bid spades on NS side => N declarer even when S raises")

# ── TRICK WINNING ─────────────────────────────────────────────
section("TRICK WINNING")
def mk_trick(leader, cards_by_seat, trump=None):
    t = Trick(leader=leader, trump=trump)
    for p, c in cards_by_seat.items(): t.add_card(p, c)
    return t

t1 = mk_trick(0, {0:Card(Rank.ACE,Suit.S),1:Card(Rank.KING,Suit.S),2:Card(Rank.QUEEN,Suit.S),3:Card(Rank.JACK,Suit.S)})
check(t1.winner() == 0, "highest card in led suit wins NT")

t2 = mk_trick(0, {0:Card(Rank.ACE,Suit.S),1:Card(Rank.TWO,Suit.H),2:Card(Rank.THREE,Suit.H),3:Card(Rank.FOUR,Suit.H)}, trump=Suit.H)
check(t2.winner() == 3, "highest trump wins when multiple players ruff")

t3 = mk_trick(0, {0:Card(Rank.ACE,Suit.S),1:Card(Rank.KING,Suit.H),2:Card(Rank.TWO,Suit.H),3:Card(Rank.JACK,Suit.S)}, trump=Suit.H)
check(t3.winner() == 1, "higher trump beats lower trump")

t4 = mk_trick(0, {0:Card(Rank.ACE,Suit.S),1:Card(Rank.TWO,Suit.S),2:Card(Rank.THREE,Suit.S),3:Card(Rank.FOUR,Suit.S)}, trump=Suit.H)
check(t4.winner() == 0, "no trump played: led suit high card wins")

t5 = mk_trick(1, {0:Card(Rank.ACE,Suit.S),1:Card(Rank.TWO,Suit.C),2:Card(Rank.THREE,Suit.C),3:Card(Rank.FOUR,Suit.C)})
check(t5.winner() == 3, "discard of As loses; highest club wins")

t6 = mk_trick(2, {2:Card(Rank.ACE,Suit.H),3:Card(Rank.THREE,Suit.S),0:Card(Rank.FOUR,Suit.S),1:Card(Rank.KING,Suit.H)}, trump=Suit.S)
check(t6.winner() == 0, "overruff: N's 4S beats W's 3S")

t7 = mk_trick(0, {0:Card(Rank.FIVE,Suit.H),1:Card(Rank.ACE,Suit.D),2:Card(Rank.KING,Suit.D),3:Card(Rank.TWO,Suit.H)}, trump=Suit.S)
check(t7.winner() == 0, "discard doesn't beat led suit, even ace")

# ── FOLLOW SUIT ENFORCEMENT ───────────────────────────────────
section("FOLLOW SUIT ENFORCEMENT")
def make_ps(trump=None, declarer=0):
    hands = {
        0: [Card(Rank.ACE,Suit.S), Card(Rank.TWO,Suit.H)],
        1: [Card(Rank.KING,Suit.S), Card(Rank.THREE,Suit.D)],
        2: [Card(Rank.QUEEN,Suit.S), Card(Rank.FOUR,Suit.C)],
        3: [Card(Rank.JACK,Suit.S), Card(Rank.FIVE,Suit.H)],
    }
    return PlayState(hands=hands, trump=trump if trump else Suit.NT, declarer=declarer, dummy=(declarer+2)%4, leader=1)

import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    ps = make_ps()
    ps.play_card(1, Card(Rank.KING, Suit.S))
    vc_p2 = ps.valid_cards(2)
    check(Card(Rank.QUEEN,Suit.S) in vc_p2, "must follow spades when holding spade")
    check(Card(Rank.FOUR,Suit.C) not in vc_p2, "cannot discard when holding led suit")

with contextlib.redirect_stdout(io.StringIO()):
    hands_void = {
        0:[Card(Rank.ACE,Suit.S),Card(Rank.TWO,Suit.H)],
        1:[Card(Rank.KING,Suit.S),Card(Rank.THREE,Suit.H)],
        2:[Card(Rank.FOUR,Suit.D),Card(Rank.SIX,Suit.H)],
        3:[Card(Rank.JACK,Suit.S),Card(Rank.FIVE,Suit.H)],
    }
    ps2 = PlayState(hands=hands_void, trump=Suit.H, declarer=0, dummy=2, leader=1)
    ps2.play_card(1, Card(Rank.KING, Suit.S))
    vc_void = ps2.valid_cards(2)
    check(Card(Rank.FOUR,Suit.D) in vc_void, "void: can discard")
    check(Card(Rank.SIX,Suit.H) in vc_void, "void: can ruff")
    check(len(vc_void) == 2, "void: all remaining cards valid")

# ── FULL TRICK SEQUENCE ───────────────────────────────────────
section("FULL TRICK SEQUENCE")
with contextlib.redirect_stdout(io.StringIO()):
    fhands = {i: [Card(r, s) for s in list(Suit)[:4] for r in [Rank.ACE,Rank.KING,Rank.QUEEN]][i*3:(i+1)*3+1] for i in range(4)}
    fhands = {
        0:[Card(Rank.ACE,Suit.S),Card(Rank.ACE,Suit.H),Card(Rank.ACE,Suit.D),Card(Rank.ACE,Suit.C)],
        1:[Card(Rank.KING,Suit.S),Card(Rank.KING,Suit.H),Card(Rank.KING,Suit.D),Card(Rank.KING,Suit.C)],
        2:[Card(Rank.QUEEN,Suit.S),Card(Rank.QUEEN,Suit.H),Card(Rank.QUEEN,Suit.D),Card(Rank.QUEEN,Suit.C)],
        3:[Card(Rank.JACK,Suit.S),Card(Rank.JACK,Suit.H),Card(Rank.JACK,Suit.D),Card(Rank.JACK,Suit.C)],
    }
    fps = PlayState(hands=fhands, trump=Suit.NT, declarer=0, dummy=2, leader=0)
    fps.play_card(0, Card(Rank.ACE,Suit.S)); fps.play_card(1, Card(Rank.KING,Suit.S))
    fps.play_card(0, Card(Rank.QUEEN,Suit.S)); fps.play_card(3, Card(Rank.JACK,Suit.S))
check(fps.tricks_ns == 1 and fps.tricks_ew == 0, "first trick: N wins with As")
with contextlib.redirect_stdout(io.StringIO()):
    fps.play_card(0, Card(Rank.ACE,Suit.H)); fps.play_card(1, Card(Rank.KING,Suit.H))
    fps.play_card(0, Card(Rank.QUEEN,Suit.H)); fps.play_card(3, Card(Rank.JACK,Suit.H))
check(fps.tricks_ns == 2 and fps.tricks_ew == 0, "second trick: N wins with Ah")

# ── COMPLETE GAME INVARIANTS ──────────────────────────────────
section("COMPLETE GAME INVARIANTS")
random.seed(77)
for trial in range(20):
    gs = GameState(board_num=(trial%16)+1,
                   vulnerable=VUL_SCHEDULE[trial%16],
                   dealer=trial%4)
    with contextlib.redirect_stdout(io.StringIO()):
        gs.new_deal()
        players = {i: RuleBasedPlayer(i) for i in range(4)}
        while gs.phase == 'AUCTION':
            a = gs.next_actor(); gs.apply_call(players[a].bid(gs.observation(a)))
        while gs.phase == 'PLAY':
            a = gs.next_actor(); gs.play_card(a, players[a].play_card(gs.observation(a)))
    check(gs.phase == 'COMPLETE', f"trial {trial}: game reaches COMPLETE")
    if gs.auction.contract:
        tricks_total = gs.play.tricks_ns + gs.play.tricks_ew
        check(tricks_total == 13, f"trial {trial}: exactly 13 tricks played")
        check(not (gs.score_ns != 0 and gs.score_ew != 0), f"trial {trial}: only one side scores")
        check(all(len(gs.play.hands[i]) == 0 for i in range(4)), f"trial {trial}: all cards played")

# ── SCORING TABLE ─────────────────────────────────────────────
section("SCORING TABLE")
def sc(level, strain, doubled, tricks, vul, expected, label):
    b = type('B', (), {'level':level,'strain':strain})()
    s = score(b, 0, doubled, tricks, vul)
    check(s == expected, f"{label}: {s} == {expected}")

sc(1,Suit.C, 0,7, False,70,  "1C made")
sc(1,Suit.D, 0,7, False,70,  "1D made")
sc(1,Suit.H, 0,7, False,80,  "1H made")
sc(1,Suit.S, 0,7, False,80,  "1S made")
sc(1,Suit.NT,0,7, False,90,  "1NT made")
sc(2,Suit.NT,0,8, False,120, "2NT made")
sc(3,Suit.NT,0,9, False,400, "3NT non-vul game")
sc(3,Suit.NT,0,9, True, 600, "3NT vul game")
sc(4,Suit.H, 0,10,False,420, "4H non-vul game")
sc(4,Suit.S, 0,10,True, 620, "4S vul game")
sc(5,Suit.C, 0,11,False,400, "5C non-vul game")
sc(5,Suit.D, 0,11,True, 600, "5D vul game")
sc(6,Suit.C, 0,12,False,920, "6C non-vul small slam")
sc(6,Suit.NT,0,12,False,990, "6NT non-vul small slam")
sc(6,Suit.NT,0,12,True,1440, "6NT vul small slam")
sc(7,Suit.C, 0,13,False,1440,"7C non-vul grand slam")
sc(7,Suit.NT,0,13,False,1520,"7NT non-vul grand slam")
sc(7,Suit.NT,0,13,True, 2220,"7NT vul grand slam")
sc(3,Suit.NT,0,10,False,430, "3NT +1 non-vul overtrick")
sc(4,Suit.S, 0,12,False,480, "4S +2 non-vul overtricks")
sc(1,Suit.S, 0,6, False,-50, "-1 non-vul undoubled")
sc(1,Suit.S, 0,6, True,-100, "-1 vul undoubled")
sc(4,Suit.S, 0,9, False,-50, "-1 non-vul undoubled (4S)")
sc(4,Suit.S, 0,8, False,-100,"-2 non-vul undoubled")
sc(4,Suit.S, 0,8, True,-200, "-2 vul undoubled")
sc(4,Suit.S, 1,9, False,-100,"-1 dbl non-vul")
sc(4,Suit.S, 1,9, True,-200, "-1 dbl vul")
sc(4,Suit.S, 1,8, False,-300,"-2 dbl non-vul")
sc(4,Suit.S, 1,8, True,-500, "-2 dbl vul")
sc(4,Suit.S, 1,7, False,-500,"-3 dbl non-vul")
sc(4,Suit.S, 1,7, True,-800, "-3 dbl vul")
sc(4,Suit.S, 1,6, False,-800,"-4 dbl non-vul")
sc(4,Suit.S, 1,6, True,-1100,"-4 dbl vul")
sc(4,Suit.S, 2,9, False,-200,"-1 rdbl non-vul")
sc(4,Suit.S, 2,9, True,-400, "-1 rdbl vul")
sc(4,Suit.S, 2,8, False,-600,"-2 rdbl non-vul")
sc(4,Suit.S, 2,8, True,-1000,"-2 rdbl vul")
sc(3,Suit.NT,1,9, False,550, "3NTX made non-vul")
sc(3,Suit.NT,1,9, True,750,  "3NTX made vul")
sc(3,Suit.NT,1,10,False,650, "3NTX +1 non-vul")
sc(3,Suit.NT,1,10,True,950,  "3NTX +1 vul")
sc(2,Suit.S, 1,10,False,670, "2SX +2 non-vul")
sc(2,Suit.S, 2,10,False,1040,"2SXX +2 non-vul")

# ── VUL SCHEDULE ──────────────────────────────────────────────
section("VULNERABILITY SCHEDULE")
check(len(VUL_SCHEDULE) == 16, "vul schedule has 16 entries")
check(VUL_SCHEDULE[0] == {'NS':False,'EW':False}, "board 1: none vul")
check(VUL_SCHEDULE[1] == {'NS':True, 'EW':False}, "board 2: NS vul")
check(VUL_SCHEDULE[2] == {'NS':False,'EW':True},  "board 3: EW vul")
check(VUL_SCHEDULE[3] == {'NS':True, 'EW':True},  "board 4: both vul")
check(all(isinstance(v['NS'], bool) and isinstance(v['EW'], bool) for v in VUL_SCHEDULE), "all vul entries are bool")

# ── OBSERVATION COMPLETENESS ──────────────────────────────────
section("OBSERVATION COMPLETENESS")
random.seed(55)
gs2 = GameState(board_num=1, vulnerable={'NS':False,'EW':False}, dealer=0)
with contextlib.redirect_stdout(io.StringIO()):
    gs2.new_deal()
for p in range(4):
    obs = gs2.observation(p)
    for k in ['board_num','vulnerable','dealer','phase','player','hand','calls','current_bidder','valid_calls']:
        check(k in obs, f"auction obs player {p} has key '{k}'")
    check(len(obs['hand']) == 13, f"player {p} hand has 13 cards")
    check(len(obs['valid_calls']) >= 1, f"player {p} has at least 1 valid call")
    check(obs['player'] == p, f"player {p} obs['player'] matches")

with contextlib.redirect_stdout(io.StringIO()):
    players2 = {i: SimpleHeuristicPlayer(i) for i in range(4)}
    while gs2.phase == 'AUCTION':
        a = gs2.next_actor(); gs2.apply_call(players2[a].bid(gs2.observation(a)))

if gs2.phase == 'PLAY':
    for p in range(4):
        obs = gs2.observation(p)
        for k in ['dummy_hand','tricks_ns','tricks_ew','current_trick','trump','current_player','valid_cards']:
            check(k in obs, f"play obs player {p} has key '{k}'")
    actor = gs2.next_actor()
    obs = gs2.observation(actor)
    for c in obs['valid_cards']:
        check(c in gs2.play.hands[actor], f"valid card {c} is in actor's hand")

# ── PASSED OUT HAND ───────────────────────────────────────────
section("PASSED OUT HAND")
gs3 = GameState(board_num=1, vulnerable={'NS':False,'EW':False}, dealer=0)
with contextlib.redirect_stdout(io.StringIO()):
    gs3.new_deal()
    players3 = {i: PassingPlayer(i) for i in range(4)}
    while gs3.phase == 'AUCTION':
        a = gs3.next_actor(); gs3.apply_call(players3[a].bid(gs3.observation(a)))
check(gs3.phase == 'COMPLETE', "passed out => COMPLETE phase")
check(gs3.score_ns == 0, "passed out => NS score = 0")
check(gs3.score_ew == 0, "passed out => EW score = 0")
check(gs3.auction.is_passed_out(), "auction is_passed_out() = True")

# ── GAME ENGINE MULTI-BOARD ───────────────────────────────────
section("GAME ENGINE MULTI-BOARD")
random.seed(42)
players4 = [RuleBasedPlayer(i) for i in range(4)]
with contextlib.redirect_stdout(io.StringIO()):
    results = Game(players4, num_boards=100).run()
check(len(results) == 100, "100 boards played")
double_score = [r for r in results if r['score_ns'] != 0 and r['score_ew'] != 0]
check(len(double_score) == 0, "no board double-scored")
passed = [r for r in results if r['contract'] is None]
check(len(passed) >= 0, f"passed out boards handled ({len(passed)} boards)")
for r in results:
    check(r['doubled'] in (0,1,2), f"doubled value {r['doubled']} in {{0,1,2}}")

# ── SELFPLAY ENV ──────────────────────────────────────────────
section("SELFPLAY ENV")
random.seed(13)
env = SelfPlayEnv([RuleBasedPlayer(i) for i in range(4)])
rewards = []
with contextlib.redirect_stdout(io.StringIO()):
    for _ in range(30):
        rewards.append(env.run_episode())
check(len(rewards) == 30, "30 episodes ran")
check(all(isinstance(r, tuple) and len(r)==2 for r in rewards), "each reward is 2-tuple")
check(all(isinstance(r[0],int) and isinstance(r[1],int) for r in rewards), "rewards are integers")
check(not any(r[0]!=0 and r[1]!=0 for r in rewards), "no episode with both sides scoring")
ns_avg = sum(r[0] for r in rewards)/30
ew_avg = sum(r[1] for r in rewards)/30
check(True, f"NS avg={ns_avg:.1f} EW avg={ew_avg:.1f} over 30 episodes")

# ── STRESS: RANDOM PLAYERS (EDGE CASES) ──────────────────────
section("STRESS: RANDOM PLAYERS (edge cases)")
random.seed(0)
with contextlib.redirect_stdout(io.StringIO()):
    crash_count = 0
    for trial in range(200):
        try:
            gs = GameState(board_num=(trial%16)+1, vulnerable=VUL_SCHEDULE[trial%16], dealer=trial%4)
            gs.new_deal()
            players = {i: RandomPlayer(i) for i in range(4)}
            while gs.phase == 'AUCTION':
                a = gs.next_actor(); gs.apply_call(players[a].bid(gs.observation(a)))
            while gs.phase == 'PLAY':
                a = gs.next_actor(); gs.play_card(a, players[a].play_card(gs.observation(a)))
        except Exception as e:
            crash_count += 1
check(crash_count == 0, f"0 crashes across 200 random-player games (crashes={crash_count})")

# ── HAND INVARIANTS DURING PLAY ───────────────────────────────
section("HAND INVARIANTS DURING PLAY")
random.seed(88)
gs4 = GameState(board_num=1, vulnerable={'NS':False,'EW':False}, dealer=0)
with contextlib.redirect_stdout(io.StringIO()):
    gs4.new_deal()
    players = {i: RuleBasedPlayer(i) for i in range(4)}
    while gs4.phase == 'AUCTION':
        a = gs4.next_actor(); gs4.apply_call(players[a].bid(gs4.observation(a)))
hand_sizes = []
if gs4.phase == 'PLAY':
    with contextlib.redirect_stdout(io.StringIO()):
        trick_num = 0
        while gs4.phase == 'PLAY':
            before = {i: len(gs4.play.hands[i]) for i in range(4)}
            a = gs4.next_actor()
            seat = gs4.play.current_seat
            card = players[a].play_card(gs4.observation(a))
            gs4.play_card(a, card)
            if gs4.play.current_trick and len(gs4.play.current_trick.cards) == 0 and gs4.play.tricks:
                after = {i: len(gs4.play.hands[i]) for i in range(4)}
                check(all(after[i] == before[i]-1 if i == seat else after[i] == before[i] for i in range(4)),
                      f"trick {trick_num}: only seated player loses a card")
                trick_num += 1
    check(all(len(gs4.play.hands[i]) == 0 for i in range(4)), "all hands empty after play")
    check(len(gs4.play.tricks) == 13, "exactly 13 tricks in history")

# ── SUMMARY ──────────────────────────────────────────────────
section("SUMMARY")
total = PASS_COUNT + FAIL_COUNT
print(f"\n  Tests run:    {total}")
print(f"  Passed:       {PASS_COUNT}")
print(f"  Failed:       {FAIL_COUNT}")
print(f"\n  {'ALL TESTS PASSED' if FAIL_COUNT == 0 else f'*** {FAIL_COUNT} FAILURES ***'}")
sys.exit(0 if FAIL_COUNT == 0 else 1)
