from flask import Flask, render_template, request
from marlon.simulate import SimulationCache, run_simulation
import plotly, json

app = Flask(__name__)
ITERATION_COUNT = 1500
SIMULATION = SimulationCache()

@app.route('/')
def home():
    if SIMULATION.value is None:
        return render_template('index.html', max_sim=0, num_graphs=0, graphs=[])
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(SIMULATION.value), graphs=[])

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if request.files['attackerFile']:
        afile = request.files['attackerFile']
    else:
        print("Missing Attacker File")
        return render_template('index.html', max_sim=0, num_graphs=0, graphs=[])

    SIMULATION.value = run_simulation(ITERATION_COUNT, afile.filename)
    graphs = json.dumps(SIMULATION.value, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(SIMULATION.value), graphs=graphs)
