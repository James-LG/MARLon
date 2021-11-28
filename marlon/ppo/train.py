import gym
import numpy as np

import cyberbattle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from marlon.ppo.wrapper import CyberbattleEnvWrapper

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = gym.wrappers.FlattenObservation(env)
        return env
    set_random_seed(seed)
    return _init

def main():
    env_id = "CyberBattleToyCtf-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = gym.make(env_id)
    env = CyberbattleEnvWrapper(env)
    check_env(env)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    #env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    
    tot_reward = np.sum(env.cyber_env._CyberBattleEnv__episode_rewards)

    print('reward', tot_reward, 'valid actions', env.valid_action_count, 'invalid actions', env.invalid_action_count)

if __name__ == '__main__':
    main()