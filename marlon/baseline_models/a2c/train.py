import gym

import cyberbattle
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper
from marlon.baseline_models.ppo.eval import evaluate

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ENABLE_ACTION_PENALTY = True

def train(evaluate_after=False):
    env_id = "CyberBattleToyCtf-v0"
    env = gym.make(env_id)

    env = AttackerEnvWrapper(
        env,
        max_timesteps=ENV_MAX_TIMESTEPS,
        enable_action_penalty=ENABLE_ACTION_PENALTY)

    check_env(env)

    model = A2C('MultiInputPolicy', Monitor(env), verbose=1)
    model.learn(total_timesteps=LEARN_TIMESTEPS, n_eval_episodes=LEARN_EPISODES)

    model.save('a2c.zip')

    if evaluate_after:
        evaluate(max_timesteps=ENV_MAX_TIMESTEPS)

if __name__ == '__main__':
    train(evaluate_after=True)
