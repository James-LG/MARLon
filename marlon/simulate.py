import json
from typing import List, Optional
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper
import gym

import plotly
from stable_baselines3.ppo.ppo import PPO
import torch
import numpy as np

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

class SimulationCache:
    value: Optional[List[str]] = None

def generate_graph_json(cyberbattle_env: CyberBattleEnv, iteration, current_score):
    fig = cyberbattle_env.render_as_fig()

    graph_json = json.dumps((fig, iteration, current_score), cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json

def run_simulation(iteration_count, agent_file):
    if agent_file.endswith('zip'):
        return run_baselines_simulation(iteration_count, agent_file)
    else:
        return run_cyberbattle_simulation(iteration_count, agent_file)

def run_baselines_simulation(iteration_count, agent_file):
    # Load the Gym environment
    gymid = "CyberBattleToyCtf-v0"
    gym_env = gym.make(gymid)
    gym_env = AttackerEnvWrapper(gym_env, enable_action_penalty=False)

    model = PPO.load(agent_file)

    obs = gym_env.reset()

    simulation = [generate_graph_json(gym_env, 0, 0)]
    current_score = 0
    for iteration in range(iteration_count):
        action, _states = model.predict(obs)
        obs, reward, done, _info = gym_env.step(action)

        assert np.shape(reward) == ()

        current_score += reward
        # If there is a jump in the reward for this step, record it for UI display.
        if done or reward > 0 or iteration == iteration_count-1:
            simulation.append(generate_graph_json(gym_env, iteration+1, current_score))
    
    return simulation

def run_cyberbattle_simulation(iteration_count, agent_file):
    # Load the Gym environment
    gymid = "CyberBattleToyCtf-v0"
    gym_env = gym.make(gymid)

    maximum_node_count = 12
    maximum_total_credentials = 10

    environment_properties = EnvironmentBounds.of_identifiers(
        maximum_node_count=maximum_node_count,
        maximum_total_credentials=maximum_total_credentials,
        identifiers=gym_env.identifiers
    )

    wrapped_env = AgentWrapper(
        gym_env,
        ActionTrackingStateAugmentation(environment_properties, gym_env.reset()))

    observation = wrapped_env.reset()

    dql_run = torch.load(agent_file)
    learner = dql_run['learner']

    simulation = [generate_graph_json(gym_env, 0, 0)]
    current_score = 0
    for iteration in range(iteration_count):

        _, gym_action, action_metadata = learner.exploit(wrapped_env, observation)
        if not gym_action:
            _, gym_action, action_metadata = learner.explore(wrapped_env)

        observation, reward, done, info = wrapped_env.step(gym_action)
        learner.on_step(wrapped_env, observation, reward, done, info, action_metadata)
        assert np.shape(reward) == ()

        current_score += reward
        # If there is a jump in the reward for this step, record it for UI display.
        if done or reward != 0 or iteration == iteration_count-1:
            simulation.append(generate_graph_json(gym_env, iteration+1, current_score))

    return simulation

if __name__ == "__main__":
    run_simulation(2000, 'ppo.zip')
