import gym

import cyberbattle
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from cyberbattle._env.cyberbattle_env import DefenderConstraint

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.multiagent.marl_algorithm import learn

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ENABLE_ACTION_PENALTY = True

def train(evaluate_after=False):
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make( env_id,
        defender_constraint=DefenderConstraint(maintain_sla=0.80),
        defender_agent=PrototypeLearningDefender())

    attacker_wrapper = AttackerEnvWrapper(
        env,
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    defender_wrapper = DefenderEnvWrapper(
        env,
        attacker_reward_store=AttackerEnvWrapper(env),
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    attacker = PPO('MultiInputPolicy', Monitor(attacker_wrapper), verbose=1)
    defender = PPO('MultiInputPolicy', Monitor(defender_wrapper), verbose=1)

    learn(attacker, defender, total_timesteps=LEARN_TIMESTEPS, n_eval_episodes=LEARN_EPISODES)

    attacker.save('ppo_attacker.zip')
    defender.save('ppo_defender.zip')

    # if evaluate_after:
    #     evaluate(max_timesteps=ENV_MAX_TIMESTEPS)

if __name__ == '__main__':
    train(evaluate_after=True)
