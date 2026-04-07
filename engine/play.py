"""Bridge play phase: trick-taking logic and play-state management.

Seats are represented as integers 0-3 mapping to N, E, S, W respectively.
The module supports both human and ML/RL agents.  The key design decision is
that the *declarer* acts on behalf of the *dummy* hand: ``current_player``
returns the declarer whenever ``current_seat`` is the dummy seat, so an agent
only ever needs to monitor ``current_player`` to know when it is its turn.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .card import Card, Suit, Rank

@dataclass
class Trick:
    """A single bridge trick.

    Tracks which seat played which card and determines the winner according to
    standard bridge rules: highest card of the led suit wins unless a trump
    card was played, in which case the highest trump wins.

    Attributes:
        leader: Seat index (0-3) of the player who leads to this trick.
        cards: Mapping of seat index to the card played by that seat.
        trump: The trump suit for this hand, or ``None`` for no-trump.
    """

    leader: int
    cards: Dict[int, Card] = field(default_factory=dict)
    trump: Optional[Suit] = None

    def led_suit(self):
        """Return the suit led to this trick, or ``None`` if no card has been played yet."""
        if not self.cards: return None
        return self.cards[self.leader].suit

    def add_card(self, player, card):
        """Record a card played by *player* into this trick.

        Args:
            player: Seat index (0-3) of the player contributing the card.
            card: The ``Card`` instance being played.
        """
        self.cards[player] = card

    def winner(self):
        """Determine the seat that wins this trick.

        Follows standard bridge rules evaluated in clockwise play order
        starting from the leader:

        * A card of the led suit beats any earlier card of the same suit only
          if its rank is strictly higher.
        * A trump card (any rank) beats any non-trump card regardless of rank.
          If multiple trumps are played, the highest trump wins.
        * No-trump hands have ``self.trump = None``; in that case only led-suit
          rank comparisons apply.

        Returns:
            The seat index (0-3) of the winning player.

        Raises:
            AssertionError: If fewer than four cards have been played.
        """
        assert len(self.cards) == 4
        led = self.led_suit()
        best_player = self.leader
        best_card = self.cards[self.leader]
        order = [(self.leader + i) % 4 for i in range(4)]
        for p in order[1:]:
            c = self.cards[p]
            if c.suit == best_card.suit:
                if c.rank > best_card.rank:
                    best_card, best_player = c, p
            elif self.trump and c.suit == self.trump:
                best_card, best_player = c, p
        return best_player

    def __repr__(self):
        seats = ['N','E','S','W']
        return ' '.join(f"{seats[p]}:{c}" for p,c in sorted(self.cards.items()))

@dataclass
class PlayState:
    """Complete mutable state for the play phase of a bridge hand.

    **Seat vs. player distinction (important for ML/RL)**

    ``current_seat`` is the *physical* seat whose turn it is to play a card
    next, advancing clockwise from the trick leader.  ``current_player`` is
    the *actor* who must supply that card: it equals the declarer whenever
    ``current_seat`` is the dummy seat, and equals ``current_seat`` otherwise.
    This indirection means an ML/RL agent representing the declarer side only
    needs to watch ``current_player`` — it will be called twice in a row
    whenever the dummy leads or plays into a trick.

    Attributes:
        hands: Mapping of seat index to the list of cards still in that seat's
            hand.  Cards are removed as they are played.
        trump: The trump suit, or ``Suit.NT`` for no-trump.
        declarer: Seat index of the declarer.
        dummy: Seat index of the dummy (partner of the declarer).
        leader: Seat index that leads to the *first* trick (normally the seat
            to the left of the declarer).
        tricks: Ordered list of completed ``Trick`` objects.
        current_trick: The trick currently in progress, or ``None`` when all
            13 tricks have been played.
        tricks_ns: Number of tricks won by the North-South partnership so far.
        tricks_ew: Number of tricks won by the East-West partnership so far.
    """

    hands: Dict[int, List[Card]]
    trump: Optional[Suit]
    declarer: int
    dummy: int
    leader: int
    tricks: List[Trick] = field(default_factory=list)
    current_trick: Optional[Trick] = None
    tricks_ns: int = 0
    tricks_ew: int = 0

    def __post_init__(self):
        self.current_trick = Trick(leader=self.leader, trump=self.trump if self.trump != Suit.NT else None)
        print(f"\n[PLAY] Trump: {self.trump} | Declarer: {['N','E','S','W'][self.declarer]} | Dummy: {['N','E','S','W'][self.dummy]} | Leader: {['N','E','S','W'][self.leader]}")
        for p, h in self.hands.items():
            print(f"  {['N','E','S','W'][p]}: {sorted(h, key=lambda c: (c.suit, c.rank))}")

    @property
    def current_seat(self):
        """The physical seat that must play the next card, in clockwise order.

        Iterates through the four seats starting from the trick leader and
        returns the first seat that has not yet contributed a card to the
        current trick.

        Returns:
            Seat index (0-3), or ``None`` if the current trick is complete or
            no trick is in progress.
        """
        t = self.current_trick
        if not t: return None
        order = [(t.leader + i) % 4 for i in range(4)]
        for p in order:
            if p not in t.cards:
                return p
        return None

    @property
    def current_player(self):
        """The actor who must supply the next card.

        This is the core design decision for ML/RL agents: the *declarer*
        controls the dummy hand.  When ``current_seat`` is the dummy, this
        property returns the declarer's seat index so that a single agent
        represents the entire declaring side.  In all other cases it returns
        ``current_seat`` unchanged.

        Returns:
            Seat index (0-3) of the actor who should call ``play_card`` next,
            or ``None`` if no trick is in progress.
        """
        seat = self.current_seat
        if seat is None: return None
        return self.declarer if seat == self.dummy else seat

    def valid_cards(self, seat):
        """Return the legal cards that *seat* may play to the current trick.

        Follows the standard bridge obligation to follow suit: if the seat
        holds any card matching the led suit it must play one of those; only
        if it holds no card of the led suit may it play any card in its hand.
        If no suit has been led yet (i.e. *seat* is leading), all cards are
        legal.

        Args:
            seat: Seat index (0-3) whose hand is examined.

        Returns:
            A list of ``Card`` objects that are legal plays for *seat*.
        """
        hand = self.hands[seat]
        led = self.current_trick.led_suit() if self.current_trick else None
        if led is None: return hand[:]
        same_suit = [c for c in hand if c.suit == led]
        return same_suit if same_suit else hand[:]

    def play_card(self, actor, card):
        """Play *card* on behalf of *actor*, advancing the trick.

        *actor* must equal ``current_player``: when it is the dummy's turn,
        the declarer (not the dummy) is the expected actor, because the
        declarer controls both hands.  The card is validated against and then
        removed from ``current_seat``'s hand (which may differ from *actor*
        when the dummy is playing).  If playing this card completes the trick,
        ``_complete_trick`` is called automatically.

        Args:
            actor: Seat index (0-3) of the agent making the move.  Must equal
                ``current_player`` (i.e. the declarer when ``current_seat`` is
                the dummy seat; otherwise the same as ``current_seat``).
            card: The ``Card`` to play.  Must be a legal card from
                ``valid_cards(current_seat)``.

        Raises:
            AssertionError: If *actor* does not match ``current_player``, or
                if *card* is not a legal play for ``current_seat``.
        """
        seat = self.current_seat
        expected = self.declarer if seat == self.dummy else seat
        assert actor == expected, f"{['N','E','S','W'][actor]} acted but expected {['N','E','S','W'][expected]}"
        assert card in self.valid_cards(seat), f"Card {card} not valid for seat {['N','E','S','W'][seat]}"
        self.hands[seat].remove(card)
        self.current_trick.add_card(seat, card)
        tag = f"{['N','E','S','W'][seat]}{'(dummy)' if seat == self.dummy else ''}"
        print(f"  [{tag}] plays {card}")
        if len(self.current_trick.cards) == 4:
            self._complete_trick()

    def _complete_trick(self):
        """Finalise a completed trick and set up the next one.

        Determines the winner, increments the appropriate partnership trick
        count, appends the trick to ``self.tricks``, and either creates a new
        ``Trick`` with the winner as leader or sets ``current_trick`` to
        ``None`` when all 13 tricks have been played.
        """
        t = self.current_trick
        w = t.winner()
        self.tricks.append(t)
        if w % 2 == 0: self.tricks_ns += 1
        else: self.tricks_ew += 1
        print(f"  Trick {len(self.tricks)}: {t} -> {['N','E','S','W'][w]} wins | NS:{self.tricks_ns} EW:{self.tricks_ew}")
        if len(self.tricks) < 13:
            self.current_trick = Trick(leader=w, trump=t.trump)
        else:
            self.current_trick = None

    def is_complete(self):
        """Return ``True`` once all 13 tricks have been played."""
        return len(self.tricks) == 13

    def tricks_for(self, declarer_side):
        """Return the tricks won by the partnership that *declarer_side* belongs to.

        Args:
            declarer_side: Any seat index (0-3) on the side of interest.
                N/S seats are even (0, 2); E/W seats are odd (1, 3).

        Returns:
            The number of tricks won by that partnership so far.
        """
        return self.tricks_ns if declarer_side % 2 == 0 else self.tricks_ew

    def result(self):
        """Summarise the final outcome of the hand.

        Returns:
            A dict with keys ``'tricks_ns'``, ``'tricks_ew'``, and
            ``'declarer_tricks'`` (tricks won by the declaring side).
        """
        dec_tricks = self.tricks_for(self.declarer)
        return {'tricks_ns': self.tricks_ns, 'tricks_ew': self.tricks_ew, 'declarer_tricks': dec_tricks}
