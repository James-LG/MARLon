import json
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from marlon.deepq.simulate import run_simulation
import os 

app = Flask(__name__)

SIMULATION = run_simulation()

@app.route('/')
def home():
    return render_template('index.html', max_sim=len(SIMULATION) - 1)

@app.route('/sim/<int:index>')
def sim(index):
    graph_json = SIMULATION[index]
    return graph_json


app.config["UPLOADS"] = "/workspaces/MARLon/uploads"

@app.route('/upload-file', methods=['POST'])
def upload_file():
    
    if request.files['attackerFile']:
        afile = request.files['attackerFile']
    else:
        print("Missing Attacker File")
        return render_template('index.html', max_sim=len(SIMULATION) - 1)
    if request.files['defenderFile']:
        dfile = request.files['defenderFile']
    else:
        print("Missing Defender File")
        return render_template('index.html', max_sim=len(SIMULATION) - 1)

    print("Both files loaded")
    afile.save(os.path.join(app.config["UPLOADS"], afile.filename))
    dfile.save(os.path.join(app.config["UPLOADS"], dfile.filename))
    
    return render_template('index.html', max_sim=len(SIMULATION) - 1)