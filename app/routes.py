from flask import Flask, render_template, jsonify, request
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)

sys.path.append(os.path.join(parent_dir, 'birdsong'))
from models import Zilpzalp

app = Flask(__name__)

model = Zilpzalp()

# load the model from checkpoint
checkpoint_path = 'model/checkpoint.tar'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# model = checkpoint['model']
state = checkpoint['state_dict']
label_dict = checkpoint['label_dict']

#model.load_state_dict(state)
#model.eval()


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify", methods=['POST'])
def classify():
    input = request.data
    with open('temp/audio.webm', 'wb+') as f:
        f.write(input)
    # audio, sr = librosa.load('temp/audio.webm')
    # print('sr', sr)
    # print('audio', audio)
    # print('shape', audio.shape)
    # length = audio.shape[0] / sr
    # print('length', length)
    # print('-----')
    # stft = librosa.stft(audio, n_fft=2048)
    # print('stft shape', stft.shape)
    # print('columns per second', stft.shape[1] / length)
    # print(stft)
    # print('-----')
    # stft = np.abs(stft)
    # print(stft)
    # stft -= stft.min()
    # print('-----')
    # print(stft)
    # stft / stft.max()
    # print(stft)
    # print('-----')
    # print(stft.shape)
    # print('-----')
    # # TODO: Add slicing function

    # spect = librosa.feature.melspectrogram(
    #     audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256, fmin=0, fmax=12000)
    # plt.imshow(spect)
    # plt.savefig('static/images/spect.svg')
    # plt.savefig('static/images/spect.png')

    # print(spect.shape)
    # print(spect)
    # print('-----')
    # sum = np.sum(spect, axis=0)
    # print(sum.shape)
    # print(sum)
    # print('-----')
    # top_indices = sum.argsort()[-216:][::-1]
    # print(len(top_indices))
    # print(top_indices)
    # print(type(top_indices))
    # print('-----')
    # top_indices_sorted = np.sort(top_indices)
    # # print(len(top_indices_sorted))
    # print(top_indices_sorted)
    # print('-----')
    # sliced_spect = spect[:, top_indices_sorted]
    # print(sliced_spect.shape)
    # print(sliced_spect)

    # plt.figure(figsize=(4, 2))
    # plt.imshow(sliced_spect)
    # plt.savefig('static/images/sliced_spect.svg')
    # plt.savefig('static/images/sliced_spect.png')

    # slice_ = "TODO"

    # make prediction
    #encoded_pred = model(torch.tensor(slice_).float()).argmax()
    #pred = label_dict[encoded_pred]

    pred = "Plegadis falcinellus"
    return jsonify({
        'species': pred,
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Plegadis_chihi_at_Aransas.jpg',
    })

# if __name__ == "__main__":
#    app.run()
