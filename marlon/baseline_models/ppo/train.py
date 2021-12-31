import gym

import cyberbattle
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

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
    env = gym.make(env_id)
    env = AttackerEnvWrapper(env)
    check_env(env)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    model.save('ppo.zip')

if __name__ == '__main__':
    main()
