"""AI agents: bidding, card play, hand evaluation, and tunable parameters."""

from .bridge_params import BridgeParams
from .hand_eval import (
    hcp, hand_shape, total_points, distribution_points, support_points,
    suit_length, suit_quality, stopper, all_suits_stopped, rule_of_20,
    biddable_suit, quick_tricks, losing_trick_count, HandShape,
)
from .bidding_agent import StateMachineBidder
from .cardplay_agent import StateMachineCardPlayer, CardTracker
from .smart_player import SmartPlayer
from .trace import DecisionTrace, TraceLog
