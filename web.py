from flask import Flask, render_template, request
from marlon.simulate import run_simulation

app = Flask(__name__)
ITERATION_COUNT = 1500
SIMULATION = None

@app.route('/')
def home():
    if SIMULATION is None:
        return render_template('index.html', max_sim=0, num_graphs=0)
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(SIMULATION))

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json

@app.route('/upload-file', methods=['POST'])
def upload_file(): 
    global SIMULATION
    if request.files['attackerFile']:
        afile = request.files['attackerFile']
    else:
        print("Missing Attacker File")
        return render_template('index.html', max_sim=0, num_graphs=0)

    SIMULATION = run_simulation(ITERATION_COUNT, afile.filename)
    print(f"SIMULATION {afile.filename} has length of {len(SIMULATION)}")
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(SIMULATION))
