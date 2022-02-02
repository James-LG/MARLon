import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from cyberbattle._env.cyberbattle_env import DefenderConstraint
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder, BaselineMarlonAgent
from marlon.baseline_models.multiagent.marl_algorithm import learn
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder, RandomMarlonAgent
from marlon.baseline_models.ppo.train import EVALUATE_EPISODES

from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.ppo.eval_defender import evaluate

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ENABLE_ACTION_PENALTY = True
EVALUATE_EPISODES = 5

def train(evaluate_after=False):
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=RandomAgentBuilder(),
        defender_builder=BaselineAgentBuilder(
            alg_type=PPO,
            policy='MultiInputPolicy'
        ),
        attacker_enable_action_penalty=ENABLE_ACTION_PENALTY,
        defender_enable_action_penalty=ENABLE_ACTION_PENALTY
    )

    universe.learn(
        total_timesteps=LEARN_TIMESTEPS,
        n_eval_episodes=LEARN_EPISODES
    )

    universe.save(
        defender_filepath='ppo_defender.zip'
    )

    if evaluate_after:
        universe.evaluate(
            n_episodes=EVALUATE_EPISODES
        )

if __name__ == '__main__':
    train(evaluate_after=True)
