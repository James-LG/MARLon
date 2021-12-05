from cyberbattle import simulation
from flask import Flask, render_template, request
from flask.helpers import url_for
from werkzeug.utils import redirect
from marlon.simulate import run_simulation

app = Flask(__name__)
ITERATION_COUNT = 1500
simulation = None

@app.route('/')
def home():
    if simulation is None:
        return render_template('index.html', max_sim=0, num_graphs=0)
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(simulation))

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = simulation[index]
    return graph_json

@app.route('/upload-file', methods=['POST'])
def upload_file(): 
    global simulation   
    if request.files['attackerFile']:
        afile = request.files['attackerFile']
    else:
        print("Missing Attacker File")
        simulation = run_simulation(ITERATION_COUNT, 'tabularq.pkl')
        return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(simulation))

    print("SIMULATING {}".format(afile.filename))
    simulation = run_simulation(ITERATION_COUNT, afile.filename)
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(simulation))