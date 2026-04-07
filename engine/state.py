"""Top-level game state for a single board of contract bridge.

Orchestrates the four phases of a board in order:
    DEAL -> AUCTION -> PLAY -> COMPLETE

Typical usage::

    state = GameState(board_num=1, dealer=0)
    state.new_deal()
    state.apply_call(some_bid)   # repeat until auction complete
    state.play_card(player, card)  # repeat until play complete
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .card import Card, Suit, deal
from .auction import AuctionState, Bid, PASS
from .play import PlayState

SEATS = ['N', 'E', 'S', 'W']
PARTNERSHIPS = {0: 'NS', 1: 'EW', 2: 'NS', 3: 'EW'}

@dataclass
class GameState:
    """Manages the full lifecycle of a single contract bridge board.

    A board progresses through four phases in strict order:

    * **DEAL** – initial state before cards are distributed.
    * **AUCTION** – players make calls until the auction is complete or
      passed out.
    * **PLAY** – thirteen tricks are played; dummy's hand is exposed.
    * **COMPLETE** – the board is scored and no further actions are taken.

    Seat indices follow the standard compass order: 0=N, 1=E, 2=S, 3=W.
    Partnerships are NS (seats 0, 2) and EW (seats 1, 3).

    Attributes:
        board_num: Sequential board identifier, used for display only.
        vulnerable: Mapping of partnership string ('NS' or 'EW') to
            whether that partnership is vulnerable this board.
        dealer: Seat index (0–3) of the dealer for this board.
        hands: Mapping of seat index to the list of cards held by that
            seat.
        auction: Active ``AuctionState``, populated after ``new_deal``.
        play: Active ``PlayState``, populated after the auction concludes
            with a contract.
        phase: Current phase string: one of 'DEAL', 'AUCTION', 'PLAY',
            or 'COMPLETE'.
        score_ns: Cumulative score accrued by the NS partnership across
            all completed boards.
        score_ew: Cumulative score accrued by the EW partnership across
            all completed boards.
    """

    board_num: int = 1
    vulnerable: Dict[str, bool] = field(default_factory=lambda: {'NS': False, 'EW': False})
    dealer: int = 0
    hands: Dict[int, List[Card]] = field(default_factory=dict)
    auction: Optional[AuctionState] = None
    play: Optional[PlayState] = None
    phase: str = 'DEAL'
    score_ns: int = 0
    score_ew: int = 0

    def new_deal(self):
        """Deal cards to all four seats and advance the phase to AUCTION.

        Populates ``self.hands`` with a freshly shuffled deal, creates a
        new ``AuctionState`` seeded with the current dealer, and transitions
        ``self.phase`` from 'DEAL' to 'AUCTION'.  Prints a summary of each
        hand and its high-card point count to stdout.

        Returns:
            GameState: Returns ``self`` to allow method chaining.
        """
        self.hands = deal()
        self.auction = AuctionState(dealer=self.dealer)
        self.phase = 'AUCTION'
        print(f"\n{'='*60}")
        print(f"[DEAL] Board {self.board_num} | Dealer:{SEATS[self.dealer]} | Vul:{self.vulnerable}")
        for p in range(4):
            hcp = sum(c.hcp() for c in self.hands[p])
            print(f"  {SEATS[p]}: {sorted(self.hands[p], key=lambda c:(c.suit,c.rank))} HCP:{hcp}")
        return self

    def apply_call(self, call: Bid):
        """Apply a call from the current bidder and advance the auction.

        Forwards ``call`` to the underlying ``AuctionState``.  When the
        auction completes with a contract, transitions automatically to
        'PLAY' via ``_start_play``.  If the auction is passed out (four
        consecutive passes), transitions to 'COMPLETE' with no score.

        Args:
            call: The ``Bid`` (or ``PASS``, ``DOUBLE``, ``REDOUBLE``)
                submitted by the current bidder.

        Raises:
            AssertionError: If the current phase is not 'AUCTION'.
        """
        assert self.phase == 'AUCTION'
        player = self.auction.current_player
        print(f"  [BID] {SEATS[player]}: {call}")
        self.auction.apply_call(call)
        if self.auction.is_complete():
            if self.auction.is_passed_out():
                print("[RESULT] Passed out - no score")
                self.phase = 'COMPLETE'
            else:
                self._start_play()

    def _start_play(self):
        """Initialise ``PlayState`` from the auction result and enter 'PLAY'.

        Reads the winning contract, declarer, and double status from the
        completed ``AuctionState``, constructs a ``PlayState`` with copies
        of all four hands, and stashes contract metadata on
        ``self._contract_meta`` for use by ``_finalize``.

        This method is called automatically by ``apply_call`` and should
        not be invoked directly.
        """
        r = self.auction.result()
        contract = r['contract']
        declarer = r['declarer']
        doubled = r['doubled']
        trump = contract.strain
        dummy = (declarer + 2) % 4
        leader = (declarer + 1) % 4
        vul = self.vulnerable[PARTNERSHIPS[declarer]]
        print(f"\n[CONTRACT] {contract}{'X'*doubled} by {SEATS[declarer]} | Vul:{vul}")
        play_hands = {p: list(h) for p, h in self.hands.items()}
        self.play = PlayState(
            hands=play_hands, trump=trump, declarer=declarer,
            dummy=dummy, leader=leader
        )
        self.phase = 'PLAY'
        self._contract_meta = (contract, declarer, doubled, vul)

    def play_card(self, player: int, card: Card):
        """Play a card on behalf of ``player`` and advance the play state.

        Delegates directly to ``PlayState.play_card``.  When all thirteen
        tricks have been played, calls ``_finalize`` to score the board and
        transition to 'COMPLETE'.

        Args:
            player: Seat index (0–3) of the player playing the card.
                When dummy is on lead, declarer supplies the card and
                passes dummy's seat index as ``player``.
            card: The ``Card`` instance to play.  Must be a legal card
                for the current trick as validated by ``PlayState``.

        Raises:
            AssertionError: If the current phase is not 'PLAY'.
        """
        assert self.phase == 'PLAY'
        self.play.play_card(player, card)
        if self.play.is_complete():
            self._finalize()

    def _finalize(self):
        """Score the completed board and update cumulative partnership totals.

        Retrieves the declarer's trick count from ``PlayState``, passes it
        to the ``scoring.score`` function along with the contract, double
        status, and vulnerability, then adds the result to ``score_ns`` if
        declarer is in the NS partnership (seats 0 or 2) or to ``score_ew``
        otherwise.  Transitions ``self.phase`` to 'COMPLETE'.

        This method is called automatically by ``play_card`` and should
        not be invoked directly.
        """
        from .scoring import score
        contract, declarer, doubled, vul = self._contract_meta
        tricks = self.play.result()['declarer_tricks']
        s = score(contract, declarer, doubled, tricks, vul)
        if declarer % 2 == 0:
            self.score_ns += s
        else:
            self.score_ew += s
        print(f"\n[FINAL] Declarer tricks: {tricks} | Score: {'+' if declarer%2==0 else '-'}{s} | NS:{self.score_ns} EW:{self.score_ew}")
        self.phase = 'COMPLETE'

    def observation(self, player: int) -> Dict[str, Any]:
        """Return an ML-ready observation dictionary for the given player.

        Produces a snapshot of all public and private information relevant
        to ``player`` at the current point in the game.  The dictionary
        structure grows as the board progresses through phases.

        **Always present:**

        * ``board_num`` – board identifier.
        * ``vulnerable`` – partnership vulnerability mapping.
        * ``dealer`` – seat index of the dealer.
        * ``phase`` – current phase string.
        * ``player`` – the requesting player's seat index.
        * ``hand`` – list of ``Card`` objects held by ``player``.

        **Present once the auction has started** (``self.auction`` is not
        ``None``):

        * ``calls`` – ordered list of all calls made so far.
        * ``current_bidder`` – seat index of the next player to call.
        * ``valid_calls`` – legal calls for the current bidder (empty list
          outside the AUCTION phase).
        * ``contract`` – current contract (may be ``None`` mid-auction).
        * ``declarer`` – seat index of declarer (may be ``None``
          mid-auction).

        **Present once play has started** (``self.play`` is not ``None``):

        * ``dummy_hand`` – full list of dummy's cards (exposed in play).
        * ``tricks_ns`` – tricks won by NS so far.
        * ``tricks_ew`` – tricks won by EW so far.
        * ``completed_tricks`` – list of all finished tricks.
        * ``current_trick`` – cards played to the trick in progress.
        * ``trump`` – trump ``Suit`` (or ``None`` for no-trump).
        * ``current_player`` – seat index of the player who must act next;
          equals declarer when dummy is on lead.
        * ``current_seat`` – seat index of the seat physically on lead
          (may be dummy).
        * ``valid_cards`` – legal cards for ``current_seat`` (not for
          ``player``).  This is intentional for the RL interface: the
          agent always receives the set of cards that are legal to play
          from the seat that is currently on lead, which may be dummy even
          when declarer is the acting player.

        Args:
            player: Seat index (0–3) of the player requesting the
                observation.

        Returns:
            Dict[str, Any]: Observation dictionary as described above.
        """
        obs = {
            'board_num': self.board_num,
            'vulnerable': self.vulnerable,
            'dealer': self.dealer,
            'phase': self.phase,
            'player': player,
            'hand': list(self.hands.get(player, [])),
        }
        if self.auction:
            obs['calls'] = list(self.auction.calls)
            obs['current_bidder'] = self.auction.current_player
            obs['valid_calls'] = self.auction.valid_calls() if self.phase == 'AUCTION' else []
            obs['contract'] = self.auction.contract
            obs['declarer'] = self.auction.declarer
        if self.play:
            dummy = self.play.dummy
            obs['dummy_hand'] = list(self.hands[dummy])
            obs['tricks_ns'] = self.play.tricks_ns
            obs['tricks_ew'] = self.play.tricks_ew
            obs['completed_tricks'] = list(self.play.tricks)
            obs['current_trick'] = self.play.current_trick
            obs['trump'] = self.play.trump
            obs['current_player'] = self.play.current_player
            obs['current_seat'] = self.play.current_seat
            obs['valid_cards'] = self.play.valid_cards(self.play.current_seat) if self.phase == 'PLAY' and self.play.current_seat is not None else []
        return obs

    def next_actor(self) -> Optional[int]:
        """Return the seat index of the player whose action is required next.

        During the AUCTION phase, returns the seat that must make the next
        call.  During the PLAY phase, returns ``PlayState.current_player``,
        which is always the declarer's seat when dummy is on lead — the
        declarer physically selects the card for dummy.  Returns ``None``
        when the board is in the 'DEAL' or 'COMPLETE' phase.

        Returns:
            Optional[int]: Seat index (0–3) of the next actor, or ``None``
            if no action is pending.
        """
        if self.phase == 'AUCTION': return self.auction.current_player
        if self.phase == 'PLAY': return self.play.current_player
        return None
