import json

import plotly
import plotly.graph_objs as go

import cyberbattle.simulation.model as model
import cyberbattle.simulation.commandcontrol as commandcontrol
import cyberbattle.samples.toyctf.toy_ctf as ctf

from flask import Flask, render_template

app = Flask(__name__)

def generate_graph_json(dbg):
    fig = go.Figure()
    traces, layout = dbg.network_as_plotly_traces()
    for t in traces:
        fig.add_trace(t)
    fig.update_layout(layout)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

def run_simulation():
    network = model.create_network(ctf.nodes)
    env = model.Environment(network=network, vulnerability_library=dict([]),identifiers=ctf.ENV_IDENTIFIERS)

    c2 = commandcontrol.CommandControl(env)
    dbg = commandcontrol.EnvironmentDebugging(c2)

    simulation = []

    c2.run_attack('client', 'SearchEdgeHistory')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'Website', 'ScanPageContent')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'GitHubProject',  'CredScanGitHistory')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'AzureStorage', 'HTTPS', 'SASTOKEN1')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'Website', 'ScanPageSource')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'Website.Directory', 'NavigateWebDirectoryFurther')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'Website.Directory', 'NavigateWebDirectory')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'Sharepoint', 'ScanSharepointParentDirectory')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'AzureResourceManager', 'HTTPS', 'ADPrincipalCreds')
    simulation.append(generate_graph_json(dbg))
    c2.run_remote_attack('client', 'AzureResourceManager', 'ListAzureResources')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'AzureVM', 'SSH', 'ReusedMySqlCred-web')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'Website', 'SSH', 'ReusedMySqlCred-web')
    simulation.append(generate_graph_json(dbg))
    c2.run_attack('Website', 'CredScanBashHistory')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('Website', 'Website[user=monitor]', 'sudo', 'monitorBashCreds')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'Website[user=monitor]', 'SSH', 'monitorBashCreds')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('Website', 'Website[user=monitor]', 'su', 'monitorBashCreds')
    simulation.append(generate_graph_json(dbg))
    c2.run_attack('Website[user=monitor]', 'CredScan-HomeDirectory')
    simulation.append(generate_graph_json(dbg))
    c2.connect_and_infect('client', 'AzureResourceManager[user=monitor]', 'HTTPS', 'azuread_user_credentials')
    simulation.append(generate_graph_json(dbg))

    return simulation

SIMULATION = run_simulation()

@app.route('/')
def home():
    return render_template('index.html', max_sim=len(SIMULATION) - 1)

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json
