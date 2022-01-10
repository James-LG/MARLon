import numpy as np

import gym
import cyberbattle
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

def evaluate(max_timesteps):
    env_id = "CyberBattleToyCtf-v0"
    cyber_env = gym.make(env_id)
    env = AttackerEnvWrapper(cyber_env, max_timesteps, enable_action_penalty=False)

    model = PPO.load('ppo.zip')

    mean_reward, std_reward = evaluate_policy(model, Monitor(env), n_eval_episodes=10)

    obs = env.reset()
    for _ in range(2000):
        action, _states = model.predict(obs)
        obs, _reward, _done, _info = env.step(action)

    tot_reward = np.sum(env.rewards)

    print('tot reward', tot_reward, 'mean reward', mean_reward, 'std reward', std_reward)
    print('valid actions', env.valid_action_count, 'invalid actions', env.invalid_action_count)

if __name__ == "__main__":
    evaluate(2000)
