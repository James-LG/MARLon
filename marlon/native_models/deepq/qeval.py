import sys
import logging
import torch
import gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

def evaluate():
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

    gymid = "CyberBattleToyCtf-v0"
    env_size = None
    iteration_count = 1500
    eval_episode_count = 10
    maximum_node_count = 12
    maximum_total_credentials = 10

    # Load the Gym environment
    if env_size:
        gym_env = gym.make(gymid, size=env_size)
    else:
        gym_env = gym.make(gymid)

    ep = w.EnvironmentBounds.of_identifiers(
        maximum_node_count=maximum_node_count,
        maximum_total_credentials=maximum_total_credentials,
        identifiers=gym_env.identifiers
    )

    model = torch.load('deepq.pkl')

    # Evaluate an agent that exploits the Q-function learnt above
    _ = learner.epsilon_greedy_search(
        gym_env,
        ep,
        learner=model,
        episode_count=eval_episode_count,
        iteration_count=iteration_count,
        epsilon=0.0,
        epsilon_minimum=0.00,
        render=False,
        plot_episodes_length=False,
        verbosity=Verbosity.Quiet,
        title="Exploiting DQL"
    )

if __name__ == "__main__":
    evaluate()
