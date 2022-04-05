from stable_baselines3 import PPO
from marlon.baseline_models.multiagent.baseline_marlon_agent import LoadFileBaselineAgentBuilder
from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.random_marlon_agent import RandomAgentBuilder
from marlon.baseline_models.ppo.train_defender import DEFENDER_SAVE_PATH

EVALUATE_EPISODES = 5

def evaluate():
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=RandomAgentBuilder(),
        defender_builder=LoadFileBaselineAgentBuilder(
            alg_type=PPO,
            file_path=DEFENDER_SAVE_PATH
        ),
        attacker_invalid_action_reward_modifier=0
    )

    universe.evaluate(
        n_episodes=EVALUATE_EPISODES
    )

if __name__ == "__main__":
    evaluate()
