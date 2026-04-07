"""Player agents for a contract bridge engine.

This module defines the abstract base class for all bridge players and four
concrete implementations that range from a pure random agent to a rule-based
agent that applies standard defensive and declarer-play heuristics.

Classes:
    Player: Abstract base class that every agent must subclass.
    RandomPlayer: Chooses uniformly at random from all legal options.
    PassingPlayer: Always passes during the auction; plays randomly.
    SimpleHeuristicPlayer: Opens/responds on HCP thresholds; plays high cards.
    RuleBasedPlayer: Delegates bidding to SimpleHeuristicPlayer; applies
        sequence leads, second-hand-low, and cover-honours card-play logic.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, Any
from .card import Suit, Rank, Card
from .auction import Bid, PASS, DOUBLE, REDOUBLE, make_bid

class Player(ABC):
    """Abstract base class for all bridge player agents.

    Every agent is assigned a seat number (0-3, where 0 = North, 1 = East,
    2 = South, 3 = West) at construction time and must implement both the
    auction and card-play phases.

    Attributes:
        seat (int): This player's seat position at the table (0-3).
    """

    def __init__(self, seat: int):
        self.seat = seat

    @abstractmethod
    def bid(self, obs: Dict[str, Any]) -> Bid:
        """Select a call to make during the auction.

        Args:
            obs (Dict[str, Any]): Observation dictionary containing at minimum:
                - ``'valid_calls'`` (list[Bid]): All legal calls at this point.
                - ``'hand'`` (list[Card]): The agent's current hand.
                - ``'calls'`` (list[Bid]): All calls made so far this auction.
                - ``'dealer'`` (int): Seat number of the dealer.
                - ``'player'`` (int): Seat number of the player to act.
                - ``'contract'`` (Bid | None): Current highest contract, if any.

        Returns:
            Bid: The chosen call (a level/suit bid, PASS, DOUBLE, or REDOUBLE).
        """
        ...

    @abstractmethod
    def play_card(self, obs: Dict[str, Any]) -> Card:
        """Select a card to play from the current hand.

        Args:
            obs (Dict[str, Any]): Observation dictionary containing at minimum:
                - ``'valid_cards'`` (list[Card]): All cards that may legally be
                  played right now.
                - ``'current_trick'`` (Trick | None): The trick in progress, or
                  ``None`` if this agent is leading.
                - ``'trump'`` (Suit): The trump suit for the current contract
                  (``Suit.NT`` for no-trumps).
                - ``'declarer'`` (int | None): Seat number of the declarer.

        Returns:
            Card: The card chosen to play.
        """
        ...


class RandomPlayer(Player):
    """Player agent that selects every action uniformly at random.

    Both the auction and card-play phases are resolved by picking one element
    at random from the list of legal options supplied in the observation.
    This agent serves as a baseline and for environment stress-testing.
    """

    def bid(self, obs):
        """Choose a random legal call.

        Args:
            obs (Dict[str, Any]): Observation dict; must contain
                ``'valid_calls'`` (list[Bid]).

        Returns:
            Bid: A uniformly random element of ``obs['valid_calls']``.
        """
        return random.choice(obs['valid_calls'])

    def play_card(self, obs):
        """Play a random legal card.

        Args:
            obs (Dict[str, Any]): Observation dict; must contain
                ``'valid_cards'`` (list[Card]).

        Returns:
            Card: A uniformly random element of ``obs['valid_cards']``.
        """
        return random.choice(obs['valid_cards'])


class PassingPlayer(Player):
    """Player agent that always passes in the auction and plays cards at random.

    Useful for isolating one side of the table in tests, or for modelling a
    player who never enters the bidding but still completes the play phase.
    """

    def bid(self, obs):
        """Always return PASS regardless of hand strength or auction context.

        Args:
            obs (Dict[str, Any]): Observation dict (contents ignored).

        Returns:
            Bid: The ``PASS`` sentinel.
        """
        return PASS

    def play_card(self, obs):
        """Play a random legal card.

        Args:
            obs (Dict[str, Any]): Observation dict; must contain
                ``'valid_cards'`` (list[Card]).

        Returns:
            Card: A uniformly random element of ``obs['valid_cards']``.
        """
        return random.choice(obs['valid_cards'])


class SimpleHeuristicPlayer(Player):
    """Player agent that applies elementary bidding and card-play heuristics.

    Auction logic:
        - Opening hand (no prior bids): opens 1NT with 20-21 HCP, 2NT with
          22+ HCP, one-of-a-suit with a five-card suit, or 1NT otherwise,
          provided HCP >= ``OPEN_HCP`` (default 12).  Otherwise passes.
        - Responding hand (partner has bid, no previous call of our own):
          raises partner's suit or bids our best suit at the cheapest level
          when HCP >= ``RESP_HCP`` (default 6).  Otherwise passes.
        - All other positions: passes if legal, otherwise takes the first
          available call.

    Card-play logic:
        - On lead (no cards in the trick yet): play the highest-ranking card.
        - Following suit: play the highest card in the led suit.
        - Discarding / ruffing: play the lowest trump to ruff, or the lowest
          remaining card to discard.

    Class Attributes:
        OPEN_HCP (int): Minimum HCP required to open the bidding (default 12).
        RESP_HCP (int): Minimum HCP required to respond to partner (default 6).
    """

    OPEN_HCP = 12
    RESP_HCP = 6

    def bid(self, obs):
        """Select a call using simple HCP-based opening and response logic.

        On the first round with no partner action, the agent opens on 12+ HCP
        using NT ranges or a natural suit bid.  When responding to partner's
        opening, the agent raises or introduces the best suit on 6+ HCP.  In
        all other situations the agent passes (or takes the first legal call if
        PASS is not available).

        Args:
            obs (Dict[str, Any]): Observation dict containing ``'valid_calls'``,
                ``'hand'``, ``'calls'``, ``'dealer'``, ``'player'``, and
                ``'contract'``.

        Returns:
            Bid: The chosen call.
        """
        valid = obs['valid_calls']
        hand = obs['hand']
        hcp = sum(c.hcp() for c in hand)
        calls = obs.get('calls', [])
        dealer = obs['dealer']
        player = obs['player']
        seat_bids = [(dealer + i) % 4 for i, c in enumerate(calls) if not c.special]
        partner = (player + 2) % 4
        my_bids = [calls[i] for i, p in enumerate([(dealer+j)%4 for j in range(len(calls))]) if p == player and not calls[i].special]
        partner_bids = [calls[i] for i, p in enumerate([(dealer+j)%4 for j in range(len(calls))]) if p == partner and not calls[i].special]
        contract = obs.get('contract')

        suit_len = {s: sum(1 for c in hand if c.suit == s) for s in list(Suit)[:4]}
        best_suit = max(suit_len, key=lambda s: (suit_len[s], s))

        if not my_bids and not partner_bids:
            if hcp >= self.OPEN_HCP:
                if hcp >= 22: target = make_bid(2, Suit.NT)
                elif hcp >= 20: target = make_bid(1, Suit.NT)
                elif suit_len[best_suit] >= 5: target = make_bid(1, best_suit)
                else: target = make_bid(1, Suit.NT)
                if target in valid: return target
            return PASS

        if partner_bids and not my_bids and contract:
            if hcp >= self.RESP_HCP:
                for level in range(contract.level, contract.level + 2):
                    b = make_bid(level, best_suit)
                    if b in valid and b > contract: return b
            return PASS if PASS in valid else valid[0]

        return PASS if PASS in valid else valid[0]

    def play_card(self, obs):
        """Select a card using high-card-first heuristics.

        On lead the highest card in hand is returned.  When following suit the
        highest card in the led suit is returned.  When void in the led suit
        the agent ruffs with the lowest available trump, or discards the lowest
        card if no trumps are held (or the contract is NT).

        Args:
            obs (Dict[str, Any]): Observation dict containing ``'valid_cards'``,
                ``'current_trick'``, and ``'trump'``.

        Returns:
            Card: The chosen card to play.
        """
        valid = obs['valid_cards']
        trick = obs.get('current_trick')
        trump = obs.get('trump')
        if not trick or not trick.led_suit():
            return max(valid, key=lambda c: c.rank)
        led = trick.led_suit()
        following = [c for c in valid if c.suit == led]
        if following:
            return max(following, key=lambda c: c.rank)
        trump_cards = [c for c in valid if trump and c.suit == trump and trump != Suit.NT]
        if trump_cards:
            return min(trump_cards, key=lambda c: c.rank)
        return min(valid, key=lambda c: c.rank)


class RuleBasedPlayer(Player):
    """Player agent that combines SimpleHeuristicPlayer bidding with smarter card play.

    The auction phase is entirely delegated to a fresh ``SimpleHeuristicPlayer``
    instance so that bidding behaviour is identical to that class.

    Card-play improvements over ``SimpleHeuristicPlayer``:

    - **Opening lead**: defenders prefer to lead the top of an honour sequence
      (e.g. K from KQJ) rather than just the highest card.
    - **Second-hand low**: when following to a trick and partner is not already
      winning, the agent plays the lowest card that beats the current trick
      winner rather than the highest possible winner.
    - **Cover honours / ruff economy**: when void in the led suit the agent
      ruffs only if the defence is not already winning the trick, using the
      smallest sufficient trump.
    - **Discard**: when unable to follow or ruff usefully, the lowest card is
      discarded.
    """

    def bid(self, obs):
        """Delegate the auction decision to SimpleHeuristicPlayer.

        Args:
            obs (Dict[str, Any]): Observation dict passed unchanged to
                ``SimpleHeuristicPlayer.bid()``.

        Returns:
            Bid: The call selected by SimpleHeuristicPlayer logic.
        """
        return SimpleHeuristicPlayer(self.seat).bid(obs)

    def play_card(self, obs):
        """Select a card using rule-based defensive and declarer-play logic.

        On lead the agent searches for the top of an honour sequence to lead;
        if no sequence exists the highest card is led.  When following suit,
        the agent plays the cheapest card that wins the trick (second-hand low)
        unless partner is already winning, in which case a low card is played.
        When void in the led suit, the agent ruffs with the lowest trump if
        the defence needs to win the trick; otherwise the lowest card is
        discarded.

        Args:
            obs (Dict[str, Any]): Observation dict containing ``'valid_cards'``,
                ``'current_trick'``, ``'trump'``, and ``'declarer'``.

        Returns:
            Card: The chosen card to play.
        """
        valid = obs['valid_cards']
        trick = obs.get('current_trick')
        trump = obs.get('trump')
        trump_suit = trump if trump != Suit.NT else None
        declarer = obs.get('declarer')
        dummy = (declarer + 2) % 4 if declarer is not None else None
        is_defense = self.seat % 2 != (declarer or 0) % 2

        if not trick or not trick.led_suit():
            if is_defense:
                sequences = self._find_sequence(valid, trump_suit)
                if sequences: return sequences[0]
            return max(valid, key=lambda c: c.rank)

        led = trick.led_suit()
        current_winner = self._trick_winner(trick, trump_suit)
        partner = (self.seat + 2) % 4
        partner_winning = current_winner == partner

        following = [c for c in valid if c.suit == led]
        if following:
            high = [c for c in following if self._beats_trick(c, trick, trump_suit)]
            if high and not partner_winning:
                return min(high, key=lambda c: c.rank)
            return min(following, key=lambda c: c.rank)

        trumps = [c for c in valid if trump_suit and c.suit == trump_suit]
        if trumps and not partner_winning and is_defense:
            return min(trumps, key=lambda c: c.rank)
        return min(valid, key=lambda c: c.rank)

    def _trick_winner(self, trick, trump):
        """Determine which seat is currently winning the trick.

        Args:
            trick: A Trick object with a ``cards`` dict mapping seat to Card
                and a ``leader`` attribute indicating the seat that led.
            trump (Suit | None): The trump suit, or ``None`` for no-trumps.

        Returns:
            int | None: Seat number of the current trick winner, or ``None``
                if no cards have been played yet.
        """
        if not trick.cards: return None
        best_p = trick.leader
        best_c = trick.cards[trick.leader]
        led = trick.cards[trick.leader].suit
        for p, c in trick.cards.items():
            if c.suit == best_c.suit and c.rank > best_c.rank:
                best_c, best_p = c, p
            elif trump and c.suit == trump and best_c.suit != trump:
                best_c, best_p = c, p
        return best_p

    def _beats_trick(self, card, trick, trump):
        """Return whether *card* would win the trick if played now.

        A card wins if it is the highest card in the led suit, or if it is the
        highest trump when at least one trump has already been played (or when
        the led suit is not trumps and no trump has been played yet).

        Args:
            card (Card): The candidate card.
            trick: A Trick object with ``cards`` and ``leader`` attributes.
            trump (Suit | None): The trump suit, or ``None`` for no-trumps.

        Returns:
            bool: ``True`` if playing *card* would take the trick.
        """
        if not trick.cards: return True
        led = trick.cards[trick.leader].suit
        best = max((c for c in trick.cards.values() if c.suit == led), key=lambda c: c.rank, default=None)
        if card.suit == led and best and card.rank > best.rank: return True
        trump_best = max((c for c in trick.cards.values() if trump and c.suit == trump), key=lambda c: c.rank, default=None)
        if card.suit == trump and trump_best and card.rank > trump_best.rank: return True
        if card.suit == trump and not trump_best and led != trump: return True
        return False

    def _find_sequence(self, hand, trump):
        """Find the top card of the best honour sequence in hand for an opening lead.

        A sequence is defined as two or more consecutive ranks in the same suit.
        The trump suit is excluded because leading trumps from a sequence is
        handled separately.  Among all sequences found, the one headed by the
        highest rank is returned.

        Args:
            hand (list[Card]): The cards available to lead.
            trump (Suit | None): The trump suit to exclude, or ``None``.

        Returns:
            list[Card]: A single-element list containing the top card of the
                best sequence, or an empty list if no sequence exists.
        """
        by_suit = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)
        best = None
        for suit, cards in by_suit.items():
            if suit == trump: continue
            cards = sorted(cards, key=lambda c: c.rank, reverse=True)
            for i in range(len(cards)-1):
                if cards[i].rank - cards[i+1].rank == 1:
                    if best is None or cards[i].rank > best.rank:
                        best = cards[i]
        return [best] if best else []
