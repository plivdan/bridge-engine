"""SmartPlayer — state-machine bridge agent combining bidding and card play.

Drop-in replacement for any ``Player`` subclass.  Uses
``StateMachineBidder`` for auction decisions (Standard American system)
and ``StateMachineCardPlayer`` for card play (with card counting,
finesse detection, and tactical defense).
"""

from engine.player import Player
from engine.card import Card
from engine.auction import Bid
from .bidding_agent import StateMachineBidder
from .cardplay_agent import StateMachineCardPlayer
from .bridge_params import BridgeParams
from .trace import TraceLog


class SmartPlayer(Player):
    """Bridge agent combining state-machine bidding and card play.

    Args:
        seat: Seat index (0=N, 1=E, 2=S, 3=W).
    """

    def __init__(self, seat: int, params=None):
        super().__init__(seat)
        self.params = params or BridgeParams()
        self.bidder = StateMachineBidder(seat, params=self.params)
        self.card_player = StateMachineCardPlayer(seat, params=self.params)
        self.trace_log = TraceLog() if self.params.trace_enabled else None

    def bid(self, obs) -> Bid:
        result = self.bidder.bid(obs)
        if self.trace_log and self.bidder.last_trace:
            self.trace_log.add(self.bidder.last_trace)
        return result

    def play_card(self, obs) -> Card:
        result = self.card_player.play_card(obs)
        if self.trace_log and self.card_player.last_trace:
            self.trace_log.add(self.card_player.last_trace)
        return result
