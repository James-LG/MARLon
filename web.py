from flask import Flask, render_template

from marlon.simulate import run_simulation

app = Flask(__name__)
ITERATION_COUNT = 1500
SIMULATION = run_simulation(ITERATION_COUNT)

@app.route('/')
def home():
    return render_template('index.html', max_sim=ITERATION_COUNT, num_graphs=len(SIMULATION))

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json
