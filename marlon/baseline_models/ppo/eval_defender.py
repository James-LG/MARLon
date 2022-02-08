import numpy as np

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.env_wrappers.environment_event_source import EnvironmentEventSource
from marlon.defender_agents.defender import PrototypeLearningDefender

def evaluate(max_timesteps):
    env_id = "CyberBattleToyCtf-v0"
    cyber_env = gym.make(
        env_id,
        defender_agent=PrototypeLearningDefender())

    wrapper_coordinator = EnvironmentEventSource()
    attacker_wrapper = AttackerEnvWrapper(
        cyber_env=cyber_env,
        event_source=wrapper_coordinator,
        max_timesteps=max_timesteps,
        enable_action_penalty=False
    )

    defender_wrapper = DefenderEnvWrapper(
        cyber_env=cyber_env,
        attacker_reward_store=attacker_wrapper,
        event_source=wrapper_coordinator,
        max_timesteps=max_timesteps,
        enable_action_penalty=False
    )

    defender_model = PPO.load('ppo_defender.zip')

    mean_reward, std_reward = evaluate_policy(defender_model, Monitor(defender_wrapper), n_eval_episodes=10)

    obs = defender_wrapper.reset()
    for _ in range(2000):
        action, _states = defender_model.predict(obs)
        obs, _reward, _done, _info = defender_wrapper.step(action)

    tot_rewards = np.sum(defender_wrapper.rewards)
    print('tot reward', tot_rewards, 'mean reward', mean_reward, 'std reward', std_reward)
    print('valid actions', defender_wrapper.valid_action_count, 'invalid actions', defender_wrapper.invalid_action_count)

if __name__ == "__main__":
    evaluate(2000)