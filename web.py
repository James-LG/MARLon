from flask import Flask, render_template

from marlon.deepq.simulate import run_simulation

app = Flask(__name__)

SIMULATION, ITERATION_NUMBER = run_simulation()

@app.route('/')
def home():
    return render_template('index.html', max_sim=ITERATION_NUMBER, list_length=len(SIMULATION))

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json
