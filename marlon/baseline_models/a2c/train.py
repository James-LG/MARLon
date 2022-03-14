from stable_baselines3 import A2C

from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse

ENV_MAX_TIMESTEPS = 2000
LEARN_TIMESTEPS = 10_000
LEARN_EPISODES = 1000 # Set this to a large value to stop at LEARN_TIMESTEPS instead.
ATTACKER_ACTION_REWARD = -1
EVALUATE_EPISODES = 5
ATTACKER_SAVE_PATH = 'a2c.zip'

def train(evaluate_after=False):
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=BaselineAgentBuilder(
            alg_type=A2C,
            policy='MultiInputPolicy'
        ),
        attacker_invalid_action_reward=ATTACKER_ACTION_REWARD
    )

    universe.learn(
        total_timesteps=LEARN_TIMESTEPS,
        n_eval_episodes=LEARN_EPISODES
    )

    universe.save(
        attacker_filepath=ATTACKER_SAVE_PATH
    )

    if evaluate_after:
        universe.evaluate(
            n_episodes=EVALUATE_EPISODES
        )

if __name__ == '__main__':
    train(evaluate_after=True)
