import json
from platform import node
from typing import List, Optional, Tuple
from cv2 import grabCut
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, EnvironmentBounds
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper
import gym
import networkx as nx

from stable_baselines3 import PPO, A2C
import torch
import numpy as np
from plotly.missing_ipywidgets import FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyberbattle import model
from cyberbattle.simulation import actions, commandcontrol

from marlon.baseline_models.env_wrappers.attack_wrapper import AttackerEnvWrapper

class SimulationCache:
    value: Optional[List[str]] = None

def generate_graph_json(cyberbattle_env: CyberBattleEnv, iteration, current_score):
 
    fig, fig2 = getCompleteGraph(cyberbattle_env)

    return fig, fig2, iteration, current_score


def getCompleteGraph(cyberbattle_env: CyberBattleEnv) -> FigureWidget:
    fig = make_subplots(rows=1, cols=2)
    fig2 = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=np.array(cyberbattle_env._CyberBattleEnv__episode_rewards).cumsum(),
                                 name='cumulative reward'), row=1, col=1)
    traces, layout, layout2 = complete_network_as_plotly_traces(cyberbattle_env, xref="x2", yref="y2")
    for trace in traces:
            fig.add_trace(trace, row=1, col=2)
            fig2.add_trace(trace, row=1, col=1)
    fig.update_layout(layout)
    fig2.update_layout(layout2)
    return fig, fig2

def get_node_information(cyberbattle_env: CyberBattleEnv, node_id: model.NodeID) -> model.NodeInfo:
    """Print node information"""
    return cyberbattle_env.get_node(node_id)

def complete_network_as_plotly_traces(cyberbattle_env: CyberBattleEnv,xref: str = "x", yref: str = "y") -> Tuple[List[go.Scatter], dict]:
    env = cyberbattle_env._CyberBattleEnv__environment
    graph = env.network
    all_nodes = nx.DiGraph(graph)

    known_nodes_ids = [node_id for node_id, _, in cyberbattle_env._actuator.discovered_nodes()]
    all_nodes = [node_id for node_id in all_nodes.nodes]
    graph_nodes = nx.shell_layout(graph, [[all_nodes[0]], all_nodes[1:]])

    discovered_nodes = []
    undiscovered_nodes = []
    owned_nodes = []

    for node_id, c in graph_nodes.items():
        if node_id in known_nodes_ids:
            if get_node_information(env, node_id).agent_installed:
                owned_nodes.append((node_id, c))
            else:
                discovered_nodes.append((node_id, c))
        else:
            undiscovered_nodes.append((node_id, c))


    def edge_text(source: model.NodeID, target: model.NodeID) -> str:
        data = env.network.get_edge_data(source, target)
        name: str = data['kind'].name
        return name
    
    color_map = {actions.EdgeAnnotation.LATERAL_MOVE: 'red',
        actions.EdgeAnnotation.REMOTE_EXPLOIT: 'orange',
        actions.EdgeAnnotation.KNOWS: 'gray'}

    def edge_color(source: model.NodeID, target: model.NodeID) -> str:
        data = env.network.get_edge_data(source, target)
        if 'kind' in data:
            return color_map[data['kind']]
        return 'black'

    layout: dict = dict(title="CyberBattle simulation", font=dict(size=10), showlegend=True,
        autosize=False, width=800, height=400,
        margin=go.layout.Margin(l=2, r=2, b=15, t=35),
        hovermode='closest',
        annotations=[dict(
            ax=graph_nodes[source][0],
            ay=graph_nodes[source][1], axref=xref, ayref=yref,
            x=graph_nodes[target][0],
            y=graph_nodes[target][1], xref=xref, yref=yref,
            arrowcolor=edge_color(source, target),
            hovertext=edge_text(source, target),
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            startstandoff=10,
            standoff=10,
            align='center',
            opacity=1
        ) for (source, target) in list(graph.edges)]
            )

    layout2: dict = dict(title="CyberBattle simulation", font=dict(size=10), showlegend=True,
        autosize=False, width=800, height=400,
        margin=go.layout.Margin(l=2, r=2, b=15, t=35),
        hovermode='closest'
        )

    trace_owned_nodes = go.Scatter(
        x=[c[0] for i, c in owned_nodes],
        y=[c[1] for i, c in owned_nodes],
        mode='markers+text',
        name='owned',
        marker=dict(symbol='circle-dot',
            size=5,
            color='#D32F2E',  # red
            line=dict(color='rgb(255,0,0)', width=8)
            ),
        text=[i for i, c in owned_nodes],
        hoverinfo='text',
        textposition="bottom center"
    )
    
    trace_undiscovered_nodes = go.Scatter(
        x=[c[0] for i, c in undiscovered_nodes],
        y=[c[1] for i, c in undiscovered_nodes],
        mode='markers+text',
        name='undiscovered',
        marker=dict(symbol='circle-dot',
            size=5,
            color='#808080',  # grey
            line=dict(color='rgb(128,128,128)', width=8)
            ),
        text=[i for i, c in undiscovered_nodes],
        hoverinfo='text',
        textposition="bottom center"
    )
    

    trace_discovered_nodes = go.Scatter(
        x=[c[0] for i, c in discovered_nodes],
        y=[c[1] for i, c in discovered_nodes],
        mode='markers+text',
        name='discovered',
        marker=dict(symbol='circle-dot',
            size=5,
            color='#0e9d00',  # green
            line=dict(color='rgb(0,255,0)', width=8)
        ),
        text=[i for i, c in discovered_nodes],
        hoverinfo='text',
        textposition="bottom center"
    )
    dummy_scatter_for_edge_legend = [
        go.Scatter(
            x=[0], y=[0], mode="lines",
            line=dict(color=color_map[a]),
            name=a.name
        ) for a in actions.EdgeAnnotation]

    all_scatters = dummy_scatter_for_edge_legend + [trace_owned_nodes, trace_undiscovered_nodes, trace_discovered_nodes]
        
    return (all_scatters, layout, layout2)

def run_simulation(iteration_count, agent_file):
    if agent_file.endswith('.zip'):
        if 'ppo' in agent_file:
            model = PPO.load(agent_file)
        elif 'a2c' in agent_file:
            model = A2C.load(agent_file)
        return run_baselines_simulation(model, iteration_count)
    else:
        return run_cyberbattle_simulation(iteration_count, agent_file)

def run_baselines_simulation(model, iteration_count):
    # Load the Gym environment
    gymid = "CyberBattleToyCtf-v0"
    gym_env = gym.make(gymid)
    gym_env = AttackerEnvWrapper(gym_env, invalid_action_reward=False)

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

    getCompleteGraph(gym_env)
    return simulation

if __name__ == "__main__":
    run_simulation(2000, 'tabularq.pkl')
