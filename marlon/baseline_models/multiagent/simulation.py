from typing import List, Tuple

import numpy as np
import networkx as nx

from plotly.missing_ipywidgets import FigureWidget
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyberbattle import model
from cyberbattle.simulation import actions
from cyberbattle._env.cyberbattle_env import CyberBattleEnv

def generate_graph_json(cyberbattle_env: CyberBattleEnv, iteration, attacker_score, defender_score):
    fig, fig2 = getCompleteGraph(cyberbattle_env)

    return fig, fig2, iteration, attacker_score, defender_score

def getCompleteGraph(cyberbattle_env: CyberBattleEnv) -> FigureWidget:
    fig = make_subplots(rows=1, cols=2)
    fig2 = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=np.array(cyberbattle_env._CyberBattleEnv__episode_rewards).cumsum(),
        name='cumulative reward'), row=1, col=1)
    traces, traces2, layout, layout2 = complete_network_as_plotly_traces(cyberbattle_env, xref="x2", yref="y2")
    for trace in traces:
        fig.add_trace(trace, row=1, col=2)
        
    for trace in traces2:
        fig2.add_trace(trace, row=1, col=1)
    
    fig.update_layout(layout)
    fig2.update_layout(layout2)
    return fig, fig2

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

    trace_invisible_nodes = go.Scatter(
        x=[c[0] for i, c in undiscovered_nodes],
        y=[c[1] for i, c in undiscovered_nodes],
        mode='markers+text',
        name=' ',
        marker=dict(symbol='circle-dot',
            size=5,
            color='#808080',
            line=dict(color='rgb(229,236,246)', width=8)
            )
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

    all_scatters = dummy_scatter_for_edge_legend + [trace_owned_nodes, trace_discovered_nodes, trace_invisible_nodes]
    all_scatters2 = [trace_owned_nodes, trace_undiscovered_nodes, trace_discovered_nodes]
    
    return (all_scatters, all_scatters2, layout, layout2)

def get_node_information(cyberbattle_env: CyberBattleEnv, node_id: model.NodeID) -> model.NodeInfo:
    """Print node information"""
    return cyberbattle_env.get_node(node_id)