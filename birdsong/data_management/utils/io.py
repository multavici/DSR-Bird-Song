import librosa
import numpy as np
import json
import warnings

def load_audio(path):
    """ Audio i/o """
    audio, sr = librosa.load(path, sr=22050)
    #assert sr == self.sr
    return audio, sr

def get_signal(audio, sr, timestamps):
    """ Extract and concatenate bird vocalizations at timesteps from audio"""
    # Convert timestamps from seconds to sample indeces
    timestamps = np.round(np.array(json.loads(timestamps)) * sr).astype(np.int)
    
    indeces = np.hstack([np.arange(t[0], t[1]) for t in timestamps])
    try:
        return audio[indeces]
    except IndexError:
        # Assure that index out of bounds errors cannot occur:
        indeces = indeces[indeces < len(audio)]
        warnings.warn('Avoided index error by truncating timestamps')
        return audio[indeces]
    
def slice_spectrogram(spec, window, stride):
    # Here the window and stride are expected to be already expressed in spectrogram
    #time-axis units
    return [spec[:, i:i+window] for i in range(0, spec.shape[1]-window, stride)]

def slice_audio(audio, sr, window, stride):
    # window, stride from nanosec to s and in samples
    window = int(window/1000 * sr)
    stride = int(stride/1000 * sr)
    return [audio[i:i+window] for i in range(0, len(audio)+1-window, stride)]
