"""Core bridge engine: cards, auction, play, scoring, game orchestration."""

from .card import Card, Suit, Rank, DECK, SUIT_SYM, deal
from .auction import Bid, AuctionState, PASS, DOUBLE, REDOUBLE, make_bid
from .play import Trick, PlayState
from .scoring import score, score_rubber
from .state import GameState
from .game import Game, SelfPlayEnv, VUL_SCHEDULE
from .player import (
    Player, RandomPlayer, PassingPlayer,
    SimpleHeuristicPlayer, RuleBasedPlayer,
)
