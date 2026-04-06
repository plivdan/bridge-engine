"""SmartPlayer — state-machine bridge agent combining bidding and card play.

Drop-in replacement for any ``Player`` subclass.  Uses
``StateMachineBidder`` for auction decisions (Standard American system)
and ``StateMachineCardPlayer`` for card play (with card counting,
finesse detection, and tactical defense).
"""

from player import Player
from card import Card
from auction import Bid
from bidding_agent import StateMachineBidder
from cardplay_agent import StateMachineCardPlayer


class SmartPlayer(Player):
    """Bridge agent combining state-machine bidding and card play.

    Args:
        seat: Seat index (0=N, 1=E, 2=S, 3=W).
    """

    def __init__(self, seat: int):
        super().__init__(seat)
        self.bidder = StateMachineBidder(seat)
        self.card_player = StateMachineCardPlayer(seat)

    def bid(self, obs) -> Bid:
        return self.bidder.bid(obs)

    def play_card(self, obs) -> Card:
        return self.card_player.play_card(obs)
