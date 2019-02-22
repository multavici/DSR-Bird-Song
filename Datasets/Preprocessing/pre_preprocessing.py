import librosa
import numpy as np
import json

def load_audio(path):
    """ Audio i/o """
    audio, sr = librosa.load(path)
    #assert sr == self.sr
    return audio, sr

def get_signal(audio, sr, timestamps):
    """ Extract and concatenate bird vocalizations at timesteps from audio"""
    # Convert timestamps from seconds to sample indeces
    timestamps = np.round(np.array(json.loads(timestamps)) * sr).astype(np.int)
    r = np.arange(audio.shape[0])
    try:
        mask = (timestamps[:,0][:,None] <= r) & (timestamps[:,1][:,None] >= r)
    except IndexError:
        print(f'Issue with slicing for audio length {r} and timestamps {timestamps}')
    # Signal as concatenation of all masked sections
    signal = audio.reshape(1, -1).repeat(mask.shape[0], axis = 0)[mask]
    return signal

def slice_spectrogram(spec, window, stride):
    #TODO: Implement this!
    # Depending on spec properties, dim1 can vary!
    #window = int(self.window / 1000 * self.sr)
    #stride = int(self.stride / 1000 * self.sr)
    return [spec[:, i:i+window] for i in range(0, spec.shape[1]-window, stride)]

def slice_audio(audio, sr, window, stride):
    # window, stride from nanosec to s and in samples
    window = int(window/1000 * sr)
    stride = int(stride / 1000 * sr)
    return [audio[i:i+window] for i in range(0, len(audio)-window, stride)]