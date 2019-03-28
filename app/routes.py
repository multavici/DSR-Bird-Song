from flask import Flask, render_template, jsonify, request
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)

sys.path.append(os.path.join(parent_dir, 'birdsong'))
sys.path.append('/snap/bin')
sys.path.append('/home/ubuntu/snap')
print(sys.path)
from models import Zilpzalp
from models import Hawk

app = Flask(__name__)

# initiate the model
model = Hawk(
    time_axis=216,
    freq_axis=256,
    no_classes=100)

# load the state of  model from checkpoint
checkpoint_path = 'model/checkpoint_Hawk_27-03'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

state = checkpoint['state_dict']
label_dict = {}
reader = csv.DictReader(open('model/top100_img_codes.csv'), fieldnames=('id', 'id2', 'species'))
for row in reader:
    label_dict[int(row['id'])] = row['species']

model.load_state_dict(state)
model.eval()


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify", methods=['POST'])
def classify():
    input = request.data
    with open('temp/audio.webm', 'wb+') as f:
        f.write(input)
    audio, sr = librosa.load('temp/audio.webm')
    length = audio.shape[0] / sr

    spect = librosa.feature.melspectrogram(
        audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256, fmin=0, fmax=12000)

    # We test 2 ways of slicing the complete spectogram

    # 1: We take the columns with most signal

    colsum = np.sum(spect, axis=0)
    top_indices = colsum.argsort()[-216:][::-1]
    top_indices_sorted = np.sort(top_indices)
    slice_maxsignal = spect[:, top_indices_sorted].reshape((1, 1, 256, 216))

    # 2: We take the first 5s seconds of signal

    slice_start = spect[:, 0:216].reshape((1, 1, 256, 216))

    # 3 We take a sliding window with the most signal

    maxdensity, i_start = 0, 0
    for i in range(len(colsum) - 216):
        density = np.sum(colsum[i:i + 216])
        if density > maxdensity:
            i_start = i
    slice_maxwindow = spect[:, i_start:i_start + 216].reshape((1, 1, 256, 216))

    # make prediction

    top5_maxsignal = get_top5_prediction(slice_maxsignal)

    top5_first5s = get_top5_prediction(slice_start)

    top5_maxwindow = get_top5_prediction(slice_maxwindow)

    # pred = "Plegadis falcinellus"
    return jsonify({
        'top5_1': top5_maxsignal,
        'top5_2': top5_first5s,
        'top5_3': top5_maxwindow,
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Plegadis_chihi_at_Aransas.jpg',
    })

def get_top5_prediction(slice_):
    output = model(torch.tensor(slice_).float()).reshape(100)
    scores_raw = torch.nn.functional.softmax(output, dim=0)
    scores, indices = scores_raw.sort(descending=True)

    top5 = []
    for code, score in zip(indices[0:5].tolist(), scores[0:5].tolist()):
        top5.append(
            (label_dict[code], f'{score:.2f}')
        )
    return top5