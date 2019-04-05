from flask import Flask, render_template, jsonify, request, abort
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import uuid
import json

# Add the models directory to the path
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
sys.path.append(os.path.join(parent_dir, 'birdsong'))

#from models import Zilpzalp
#from models import Hawk
from models import LstmModel
from utils import avg_score, maxwindow_score, get_top5_prediction

app = Flask(__name__)

# Initiate the model
model = LstmModel(
    time_axis=216,
    freq_axis=256,
    no_classes=100)

# Load the state of  model from checkpoint
checkpoint_path = 'model/checkpoint_Lstm_29-03'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state = checkpoint['state_dict']
model.load_state_dict(state)
model.eval()

# Add the dictionary with the species info
label_dict = {}
reader = csv.DictReader(open('model/top100_codes_translated.csv'))
for row in reader:
    label_dict[int(row['id1'])] = {
        'name': row['english'],
        'img_source': row['img_source'],
        'img_link': row['img_link'],
        'wiki_link': row['wiki_link'],
    }


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify", methods=['POST'])
def classify():
    # Get audio data from client
    input_ = request.data
    
    # Save audio date as .webm file
    audio_path = 'temp/' + str(uuid.uuid4()) +'.webm'
    with open(audio_path, 'wb+') as f:
        f.write(input_)

    # Load audio
    try:
        audio, sr = librosa.load(audio_path)
        # os.remove(audio_path)
    except:
        # os.remove(audio_path)
        return 'error', 500
    
    # Make predictions
    scores, indices = maxwindow_score(model, audio, sr)
    top5 = get_top5_prediction(label_dict, scores, indices)

    # Add to logs
    with open('logs.txt', 'a+') as f:
        f.write(json.dumps({
            'input': audio_path,
            'prediction': top5,
        }) + '\n')

    # Return predictions to client
    return jsonify({
        'predictions': top5,
    })
