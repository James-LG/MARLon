import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from cyberbattle._env.cyberbattle_env import DefenderConstraint

def main():
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make( env_id,
                    defender_constraint=DefenderConstraint(maintain_sla=0.80),
                    defender_agent=PrototypeLearningDefender())

    env = DefenderEnvWrapper(env)
    check_env(env)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    model.save('ppo_defender.zip')

if __name__ == '__main__':
    main()
