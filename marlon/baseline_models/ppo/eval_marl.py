from stable_baselines3 import PPO
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.ppo.train_marl import ATTACKER_SAVE_PATH
from marlon.baseline_models.ppo.train_marl import DEFENDER_SAVE_PATH

EVALUATE_EPISODES = 5

def evaluate():
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=LoadFileBaselineAgentBuilder(
            alg_type=PPO,
            file_path=ATTACKER_SAVE_PATH
        ),
        defender_builder=LoadFileBaselineAgentBuilder(
            alg_type=PPO,
            file_path=DEFENDER_SAVE_PATH
        ),
        attacker_invalid_action_reward=0,
        defender_invalid_action_reward=0
    )

    universe.evaluate(
        n_episodes=EVALUATE_EPISODES
    )

if __name__ == "__main__":
    evaluate()
