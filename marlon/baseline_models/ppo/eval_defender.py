import numpy as np

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.defender_agents.defender import PrototypeLearningDefender

def evaluate(max_timesteps):
    env_id = "CyberBattleToyCtf-v0"
    cyber_env = gym.make(env_id,
                    defender_agent=PrototypeLearningDefender())
    env = DefenderEnvWrapper(cyber_env)

    model = PPO.load('ppo_defender.zip')

    mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=10)

    obs = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs)
        obs, _reward, _done, _info = env.step(action)
        
    tot_rewards = np.sum(env.rewards)
    print('tot reward', tot_rewards, 'mean reward', mean_reward, 'std reward', std_reward)
    print('valid actions', env.valid_action_count, 'invalid actions', env.invalid_action_count)

if __name__ == "__main__":
    evaluate(2000)