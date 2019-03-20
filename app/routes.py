from flask import Flask, render_template, jsonify, request
import librosa

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify", methods=['POST'])
def classify():
    audio = request.data
    with open('temp/audio.webm', 'wb+') as f:
        f.write(audio)
    y, sr = librosa.load('temp/audio.webm')
    print(y)
    # TODO: Add prediction function


    pred = "Plegadis falcinellus"
    return jsonify({'species': pred})


app.run(host='0.0.0.0', port=50000)