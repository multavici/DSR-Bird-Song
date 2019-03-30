import librosa
import torch
import numpy as np

from data_management.utils.signal_extraction import signal_noise_separation
from data_management.utils.io import slice_audio

# Define different functions to determine the score depending on the way the recording is sliced
def avg_score(model, audio, sr):
    """
    Slice the audio in overlapping slices and calculate the predictions as an 
    average of the predictions on the spectograms of the different slices
    """
    signal, _ = signal_noise_separation(audio)
    if len(signal) / sr < 5:
        raise ValueError
    slices = slice_audio(signal, sr, 5000, 2500)
    total_scores = torch.zeros(100)
    for slice_ in slices:
        sliced_spect = librosa.feature.melspectrogram(
            slice_, n_mels=256, fmin=0, fmax=12000)
        sliced_spect = sliced_spect.reshape((1, 1, 256, 216))
        output = model(torch.tensor(sliced_spect).float()).reshape(100)
        scores_raw = torch.nn.functional.softmax(output, dim=0)
        total_scores = total_scores.add(scores_raw)
    avg_scores = total_scores / len(slices)
    scores, indices = scores_raw.sort(descending=True)
    return (scores, indices)


def maxwindow_score(model, audio, sr):
    """
    Let a moving window of 5 seconds slide over a spectogram of the audio and
    choose the slice with maximum signal.
    Calculate the prediction on this slice.
    """
    spect = librosa.feature.melspectrogram(
        audio, n_mels=256, fmin=0, fmax=12000)
    colsum = np.sum(spect, axis=0)
    maxdensity, i_start = 0, 0
    for i in range(len(colsum) - 216):
        density = np.sum(colsum[i:i + 216])
        if density > maxdensity:
            maxdensity = density
            i_start = i
    slice_maxwindow = spect[:, i_start:i_start + 216].reshape((1, 1, 256, 216))
    output = model(torch.tensor(slice_maxwindow).float()).reshape(100)
    scores_raw = torch.nn.functional.softmax(output, dim=0)
    scores, indices = scores_raw.sort(descending=True)
    return (scores, indices)


def get_top5_prediction(label_dict, scores, indices):
    top5 = []
    for code, score in zip(indices[0:5].tolist(), scores[0:5].tolist()):
        top5.append([
            label_dict[code]['name'],
            label_dict[code]['img_source'],
            label_dict[code]['img_link'],
            label_dict[code]['wiki_link'],
            f'{score:.2f}'],
        )
    return top5