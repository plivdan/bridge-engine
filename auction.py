"""Bridge auction (bidding) engine.

This module models the bidding phase of a contract bridge game.  It provides:

- ``Bid`` -- an immutable value object representing a single call (a suit/NT
  bid at a given level, or one of the three special calls PASS, DOUBLE, and
  REDOUBLE).
- ``AuctionState`` -- a mutable object that tracks the full history of calls
  made during an auction, enforces the rules of legal bidding, and determines
  the final contract and declarer when the auction is complete.

Typical usage::

    state = AuctionState(dealer=0)
    state.apply_call(make_bid(1, Suit.SPADE))
    state.apply_call(PASS)
    state.apply_call(make_bid(2, Suit.SPADE))
    state.apply_call(PASS)
    state.apply_call(PASS)
    state.apply_call(PASS)
    assert state.is_complete()
    print(state.result())  # {'contract': 2♠, 'declarer': 0, 'doubled': 0}
"""

from dataclasses import dataclass, field
from typing import Optional, List
from card import Suit, SUIT_SYM

PASS_BID = 'PASS'
DOUBLE_BID = 'X'
REDOUBLE_BID = 'XX'

@dataclass(frozen=True)
class Bid:
    """An immutable bridge call (bid, pass, double, or redouble).

    ``Bid`` is a frozen dataclass so instances are hashable and safe to use as
    dictionary keys or set members.

    There are two categories of ``Bid``:

    Normal bids:
        Represent a contract-level call such as "1♠" or "3NT".  ``level`` is
        an integer in the range 1–7, ``strain`` is a ``Suit`` value (the five
        strains in ascending order are ♣ < ♦ < ♥ < ♠ < NT), and ``special``
        is ``None``.

    Special calls:
        Represent PASS, DOUBLE (``X``), or REDOUBLE (``XX``).  For these,
        ``level`` and ``strain`` are both ``None`` and ``special`` holds the
        string constant (``PASS_BID``, ``DOUBLE_BID``, or ``REDOUBLE_BID``).
        The module-level singletons ``PASS``, ``DOUBLE``, and ``REDOUBLE``
        should be used instead of constructing new instances.

    Ordering:
        Normal bids compare by ``(level, strain)`` tuple, which matches the
        natural rank order used in bidding (e.g. 1♣ < 1♦ < … < 7NT).
        Comparisons involving at least one special call always return ``False``
        for ``<`` (and derived operators), so special calls are unordered
        relative to everything else.

    Attributes:
        level: Contract level (1–7) for normal bids; ``None`` for special calls.
        strain: Trump suit or NT for normal bids; ``None`` for special calls.
        special: One of ``PASS_BID``, ``DOUBLE_BID``, ``REDOUBLE_BID``, or
            ``None`` for normal bids.
    """

    level: Optional[int]
    strain: Optional[Suit]
    special: Optional[str] = None

    def __repr__(self):
        """Return a short human-readable string such as ``'1♠'`` or ``'PASS'``."""
        if self.special: return self.special
        return f"{self.level}{SUIT_SYM[self.strain]}"

    def __lt__(self, other):
        """Return ``True`` iff both bids are normal and self ranks lower than other.

        Args:
            other: Another ``Bid`` to compare against.

        Returns:
            ``False`` if either bid is a special call (PASS/X/XX); otherwise
            ``True`` when ``(self.level, self.strain) < (other.level, other.strain)``.
        """
        if self.special or other.special: return False
        return (self.level, self.strain) < (other.level, other.strain)

    def __le__(self, other): return self == other or self < other
    def __gt__(self, other): return not self <= other
    def __ge__(self, other): return not self < other

PASS = Bid(None, None, PASS_BID)
DOUBLE = Bid(None, None, DOUBLE_BID)
REDOUBLE = Bid(None, None, REDOUBLE_BID)

def make_bid(level, strain): return Bid(level, strain)

@dataclass
class AuctionState:
    """Mutable state machine for a single bridge auction.

    An ``AuctionState`` is created at the start of a deal and updated call by
    call via ``apply_call()``.  It enforces the laws of contract bridge bidding
    and, once the auction is complete, exposes the final contract, declarer, and
    doubling level through ``result()``.

    Declarer assignment:
        When a player makes a normal bid that becomes (or updates) the current
        contract, the declarer is set to the *first* player on that partnership's
        side (i.e. the player or their partner) who called that same strain
        during the auction — not necessarily the player who made the final bid.
        This mirrors the official bridge rule and means the opening leader faces
        the correct hand.

    Doubling state:
        ``doubled`` tracks whether the current contract is undoubled (``0``),
        doubled (``1``), or redoubled (``2``).  It resets to ``0`` whenever a
        new normal bid supersedes the previous contract.

    Attributes:
        dealer: Seat index (0–3) of the player who deals and bids first.
        calls: Ordered list of every ``Bid`` made so far, oldest first.
        current_player: Seat index of the player whose turn it is to call.
        contract: The highest normal bid made so far, or ``None`` if the
            auction has not yet seen a normal bid.
        declarer: Seat index of the declarer for the current contract, or
            ``None`` if no normal bid has been made yet.
        doubled: ``0`` = undoubled, ``1`` = doubled, ``2`` = redoubled.
    """

    dealer: int
    calls: List[Bid] = field(default_factory=list)
    current_player: int = 0
    contract: Optional[Bid] = None
    declarer: Optional[int] = None
    doubled: int = 0

    def __post_init__(self):
        self.current_player = self.dealer

    def _last_bid(self):
        """Return the most recent normal (non-special) bid, or ``None``.

        Scans ``self.calls`` in reverse order and returns the first entry that
        is not a special call (PASS/X/XX).

        Returns:
            The last normal ``Bid`` in the call sequence, or ``None`` if no
            normal bid has been made yet.
        """
        for c in reversed(self.calls):
            if not c.special: return c
        return None

    def _last_doubler_side(self):
        """Return the side (0 or 1) that made the most recent DOUBLE, or ``None``.

        Scans ``self.calls`` in reverse order stopping at the most recent
        normal bid (which resets doubling context).

        Returns:
            The side index (``bidder_seat % 2``) of the player who made the
            most recent DOUBLE in the current doubling window, or ``None`` if
            the current contract has not been doubled.
        """
        for c in reversed(self.calls):
            if c.special == DOUBLE_BID: return self._bidder_of(c)
            if not c.special: return None
        return None

    def _bidder_of(self, call):
        """Return the seat index of the player who made a given call.

        Args:
            call: A ``Bid`` object that must be present in ``self.calls``.
                  Uses the *first* occurrence (via ``list.index``), so callers
                  should not pass duplicate ``Bid`` objects.

        Returns:
            Integer seat index (0–3) computed from the dealer seat and the
            call's position in the call list.
        """
        idx = self.calls.index(call)
        return (self.dealer + idx) % 4

    def valid_calls(self):
        """Return a list of all legally playable calls for the current player.

        The rules applied are:

        - PASS is always legal.
        - If no normal bid has been made yet, any of the 35 normal bids
          (levels 1–7 × five strains) is legal.
        - If a normal bid has already been made, only bids strictly higher
          than the current contract are legal.
        - DOUBLE (``X``) is legal when the current contract is undoubled
          (``doubled == 0``) and was bid by the *opposing* side.
        - REDOUBLE (``XX``) is legal when the current contract is doubled
          (``doubled == 1``) and the DOUBLE was made by the *opposing* side.

        Returns:
            A ``list`` of ``Bid`` objects, always containing at least ``PASS``.
        """
        calls = [PASS]
        lb = self._last_bid()
        if lb:
            for level in range(1, 8):
                for strain in list(Suit)[:5]:
                    b = make_bid(level, strain)
                    if b > lb: calls.append(b)
            last_bid_side = (self.dealer + [i for i,c in enumerate(self.calls) if not c.special][-1]) % 4 % 2 if any(not c.special for c in self.calls) else -1
            cur_side = self.current_player % 2
            if self.doubled == 0 and last_bid_side != cur_side:
                calls.append(DOUBLE)
            if self.doubled == 1 and self._last_doubler_side() % 2 != self.current_player % 2:
                calls.append(REDOUBLE)
        else:
            for level in range(1, 8):
                for strain in list(Suit)[:5]:
                    calls.append(make_bid(level, strain))
        return calls

    def apply_call(self, call):
        """Validate a call and advance the auction state.

        Appends the call to ``self.calls``, updates ``contract``, ``declarer``,
        ``doubled``, and ``current_player`` according to bridge rules.

        Declarer logic: when a normal bid is made, the auction scans all
        previous normal bids by players on the same side (``seat % 2``).  If
        any of those bids named the same strain, the declarer is set to the
        *earliest* such player; otherwise the declarer becomes the current
        player.  This implements the "first to bid the strain on the winning
        side" rule.

        Args:
            call: A ``Bid`` to apply.  Must be a member of ``valid_calls()``.

        Returns:
            ``self``, to allow chaining (e.g. ``state.apply_call(b1).apply_call(b2)``).

        Raises:
            AssertionError: If ``call`` is not in ``valid_calls()``.
        """
        assert call in self.valid_calls(), f"Invalid call {call}"
        self.calls.append(call)
        if call == DOUBLE: self.doubled = 1
        elif call == REDOUBLE: self.doubled = 2
        elif not call.special:
            self.doubled = 0
            self.contract = call
            strain = call.strain
            side = self.current_player % 2
            for i, c in enumerate(self.calls[:-1]):
                if not c.special and c.strain == strain and (self.dealer + i) % 4 % 2 == side:
                    self.declarer = (self.dealer + i) % 4
                    break
            if self.declarer is None:
                self.declarer = self.current_player
        self.current_player = (self.current_player + 1) % 4
        return self

    def is_complete(self):
        """Return ``True`` when the auction has ended under bridge rules.

        An auction ends in one of two ways:

        1. Four consecutive passes with no normal bid ever made (passed-out
           deal) — all four players passed on their first turn.
        2. Three consecutive passes *after* a normal bid has been made,
           confirming the final contract.

        Returns:
            ``True`` if the auction is over, ``False`` otherwise.
        """
        if len(self.calls) < 4: return False
        last4 = self.calls[-4:]
        if all(c == PASS for c in last4): return True
        if len(self.calls) >= 4 and all(c.special == PASS_BID for c in self.calls[-3:]) and self.contract:
            return True
        return False

    def is_passed_out(self):
        """Return ``True`` if the deal was passed out (no contract established).

        A deal is passed out when all four players pass on their opening call,
        i.e. the last four calls are all PASS and no normal bid was ever made.
        In this case the hand is thrown in with no score.

        Returns:
            ``True`` if the auction ended with four consecutive passes and
            ``self.contract`` is ``None``; ``False`` otherwise.
        """
        return len(self.calls) >= 4 and all(c == PASS for c in self.calls[-4:]) and self.contract is None

    def result(self):
        return {
            'contract': self.contract,
            'declarer': self.declarer,
            'doubled': self.doubled,
        }
