"""Entry point and demo runner for the bridge engine.

Showcases the engine with different player types: random, heuristic,
rule-based, and a self-play environment for ML training loops.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from player import RandomPlayer, PassingPlayer, SimpleHeuristicPlayer, RuleBasedPlayer
from game import Game, SelfPlayEnv

def demo_random(n=1):
    print("\n" + "="*60)
    print("DEMO: Random players")
    players = [RandomPlayer(i) for i in range(4)]
    g = Game(players, num_boards=n)
    return g.run()

def demo_heuristic(n=3):
    print("\n" + "="*60)
    print("DEMO: Heuristic players")
    players = [SimpleHeuristicPlayer(i) for i in range(4)]
    g = Game(players, num_boards=n)
    return g.run()

def demo_rule_based(n=3):
    print("\n" + "="*60)
    print("DEMO: Rule-based players")
    players = [RuleBasedPlayer(i) for i in range(4)]
    g = Game(players, num_boards=n)
    return g.run()

def demo_self_play_env(n=5):
    print("\n" + "="*60)
    print("DEMO: SelfPlayEnv (ML training loop)")
    players = [RuleBasedPlayer(i) for i in range(4)]
    env = SelfPlayEnv(players)
    rewards = []
    for _ in range(n):
        r = env.run_episode()
        rewards.append(r)
    print(f"\n[ML ENV] Episodes: {n} | Rewards: {rewards}")
    print(f"  Avg NS: {sum(r[0] for r in rewards)/n:.1f} | Avg EW: {sum(r[1] for r in rewards)/n:.1f}")
    return rewards

if __name__ == '__main__':
    import random
    random.seed(42)
    demo_random(1)
    random.seed(0)
    demo_heuristic(2)
    random.seed(7)
    demo_rule_based(2)
    random.seed(99)
    demo_self_play_env(3)
