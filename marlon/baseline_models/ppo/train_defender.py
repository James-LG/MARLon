import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from cyberbattle._env.cyberbattle_env import DefenderConstraint

from marlon.defender_agents.defender import PrototypeLearningDefender
from marlon.baseline_models.env_wrappers.defend_wrapper import DefenderEnvWrapper
from marlon.baseline_models.ppo.eval_defender import evaluate

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ENABLE_ACTION_PENALTY = True

def train(evaluate_after=True):
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make( env_id,
                    defender_constraint=DefenderConstraint(maintain_sla=0.80),
                    defender_agent=PrototypeLearningDefender())

    env = DefenderEnvWrapper(
        env,
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    check_env(env)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)

    model.save('ppo_defender.zip')

    if evaluate_after:
        evaluate(max_timesteps=ENV_MAX_TIMESTEPS)

if __name__ == '__main__':
    train(evaluate_after=True)
