from flask import Flask, render_template, jsonify, request
import librosa
import torch

app = Flask(__name__)

# load the model from checkpoint
# checkpoint_path = 'model/checkpoint.tar'
# checkpoint = torch.load(checkpoint_path, map_location='cpu')

# model = checkpoint['model']
# state = checkpoint['state_dict']
# label_dict = checkpoint['label_dict']

# model.load_state_dict(state)

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
    print(sr)
    # TODO: Add slicing function
    slice_ = "TODO"

    # make prediction
    #encoded_pred = model(torch.tensor(slice_).float()).argmax()
    #pred = label_dict[encoded_pred]

    pred = "Plegadis falcinellus"
    return jsonify({
        'species': pred,
        'image_url': 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Plegadis_chihi_at_Aransas.jpg',
        })

#if __name__ == "__main__":
#    app.run()