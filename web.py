from flask import Flask, render_template, request
from marlon.simulate import SimulationCache, simulate
import plotly, json

app = Flask(__name__)
ITERATION_COUNT = 1500
SIMULATION = SimulationCache()

@app.route('/')
def home():
    if SIMULATION.value is None:
        return render_template('index.html', num_graphs=0, graphs=[], attacker_name = "-", defender_name = "-")
    return render_template('index.html', num_graphs=len(SIMULATION.value), graphs=[], attacker_name = "-", defender_name = "-")

@app.route('/upload-file', methods=['POST'])
def upload_file():
    #Get Agent Configurations
    attacker_option = request.form['attackerOptions']
    defender_option = request.form['defenderOptions']

    attacker_file = None
    if 'attackerFile' in request.files:
        attacker_file = request.files['attackerFile'].filename
    defender_file = None
    if 'defenderFile' in request.files:
        defender_file = request.files['defenderFile'].filename

    SIMULATION.value = simulate(
        timesteps=ITERATION_COUNT,
        attacker_option=attacker_option,
        defender_option=defender_option,
        attacker_file=attacker_file,
        defender_file=defender_file
    )

    attacker_name = attacker_option
    if attacker_name == "Load":
        attacker_name = attacker_file
    
    defender_name = defender_option
    if defender_name == "Load":
        defender_name = defender_file

    graphs = json.dumps(SIMULATION.value, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', num_graphs=len(SIMULATION.value), graphs=graphs, attacker_name = attacker_name, defender_name = defender_name)

if __name__ == "__main__":
    app.run(host='localhost', port=5000)