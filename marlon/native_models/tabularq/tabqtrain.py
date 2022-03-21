import sys
import logging
import torch
import gym
from cyberbattle.agents.baseline import learner
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_tabularqlearning as tabularq
from cyberbattle.agents.baseline.agent_wrapper import Verbosity


def train():
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

    gymid = "CyberBattleToyCtf-v0"
    env_size = None
    iteration_count = 1500
    training_episode_count = 20
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

    # Evaluate a Tabular Q-learning agent
    tabularq_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=tabularq.QTabularLearner(
        ep,
        gamma=0.015, learning_rate=0.01, exploit_percentile=100),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.01,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="Tabular Q-learning"
)

    torch.save(tabularq_run['learner'], 'tabularq.pkl')

if __name__ == "__main__":
    train()
