import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from marlon.defender_agents.defender import prototype_learning_defender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from cyberbattle._env.cyberbattle_env import DefenderConstraint

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
    env = gym.make( env_id,
                    defender_constraint=DefenderConstraint(maintain_sla=0.80),
                    defender_agent=prototype_learning_defender())

    env = DefenderEnvWrapper(env)
    check_env(env)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    model.save('ppo_defender.zip')

if __name__ == '__main__':
    main()
