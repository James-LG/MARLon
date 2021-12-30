import numpy as np

import gym
import cyberbattle
from stable_baselines3.ppo.ppo import PPO

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper


def main():
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make(env_id)
    env = AttackerEnvWrapper(env)

    model = PPO.load('ppo.zip')

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, _reward, _done, _info = env.step(action)
    
    tot_reward = np.sum(env.cyber_env._CyberBattleEnv__episode_rewards)

    print('reward', tot_reward, 'valid actions', env.valid_action_count, 'invalid actions', env.invalid_action_count)

if __name__ == "__main__":
    main()
