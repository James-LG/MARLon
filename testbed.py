
from stable_baselines3 import A2C, PPO
import torch
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder, LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.qcompat_agent import QCompatibilityAgentBuilder
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder


def main():
    attacker_builder = RandomAgentBuilder()
    # attacker_builder = QCompatibilityAgentBuilder(
    #     learner=torch.load('deepq.pkl')
    # )
    # attacker_builder = LoadFileBaselineAgentBuilder(A2C, 'ppo_marl_attacker.zip')

    #defender_builder = LoadFileBaselineAgentBuilder(A2C, 'ppo_marl_defender.zip')
    #defender_builder = RandomAgentBuilder()
    defender_builder = None

    universe = MultiAgentUniverse.build(
        attacker_builder=attacker_builder,
        attacker_invalid_action_reward=0,
        defender_builder=defender_builder,
        defender_invalid_action_reward=0,
    )

    universe.evaluate(5)

if __name__ == "__main__":
    main()