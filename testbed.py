
from stable_baselines3 import A2C, PPO
import torch
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder, LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.qcompat_agent import QCompatibilityAgentBuilder
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder


def main():
    attackers = {
        'random': RandomAgentBuilder(),
        'tabularq': QCompatibilityAgentBuilder('tabularq.pkl'),
        'deepq': QCompatibilityAgentBuilder('deepq.pkl'),
        'ppo': LoadFileBaselineAgentBuilder(PPO, 'ppo.zip'),
        'ppo_marl': LoadFileBaselineAgentBuilder(PPO, 'ppo_marl_attacker.zip'),
        'a2c': LoadFileBaselineAgentBuilder(A2C, 'a2c.zip'),
        'a2c_marl': LoadFileBaselineAgentBuilder(A2C, 'a2c_marl_attacker.zip'),
    }

    defenders = {
        'none': None,
        'random': RandomAgentBuilder(),
        'ppo': LoadFileBaselineAgentBuilder(PPO, 'ppo_defender.zip'),
        'ppo_marl': LoadFileBaselineAgentBuilder(PPO, 'ppo_marl_defender.zip'),
        'a2c': LoadFileBaselineAgentBuilder(A2C, 'a2c_defender.zip'),
        'a2c_marl': LoadFileBaselineAgentBuilder(A2C, 'a2c_marl_defender.zip'),
    }

    universe = MultiAgentUniverse.build(
        attacker_builder=attackers['random'],
        attacker_invalid_action_reward_modifier=0,
        defender_builder=defenders['none'],
        defender_invalid_action_reward_modifier=0,
    )

    universe.evaluate(5)

if __name__ == "__main__":
    main()