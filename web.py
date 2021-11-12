from flask import Flask, render_template

from marlon.deepq.simulate import run_simulation

app = Flask(__name__)

SIMULATION = run_simulation()

@app.route('/')
def home():
    return render_template('index.html', max_sim=len(SIMULATION) - 1)

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json
