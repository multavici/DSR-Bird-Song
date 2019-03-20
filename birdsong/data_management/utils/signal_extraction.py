"""
The following functions serve to highlight sections of bird vocalizations in a
audio recording. They are based on the methodologies described in Lasseck 2013 
and Sprengler 2017.

Usage:
import the function "signal_timestamps" from this script and pass it a path to
a audio file. It will then return the total duration, the summed duration of 
foreground bird-vocalizations and a json of start and stop timestamps for
sections with bird vocalizations. 
"""

import librosa
import librosa.display
import numpy as np
from scipy.ndimage import morphology
import json

def normalized_stft(audio):
    """ Short-time Fourier Transform with hann window 2048 and 75% overlap (default),
    normalized into 0-1 scale. """
    stft = np.abs(librosa.stft(audio, n_fft=2048))  # 2048, win_length = 512))
    stft -= stft.min()
    return stft / stft.max()


def median_mask(spec, threshold, inv=False):
    """ Returns a binary mask for values that are above a threshold times the row
    and column medians. Inverting the mask is possible. """
    row_medians = np.expand_dims(np.median(spec, axis=1), axis=1)
    col_medians = np.expand_dims(np.median(spec, axis=0), axis=0)
    # Broadcasting the median vectors over the spectrogram:
    if inv:
        mask = np.where((spec > threshold * row_medians) &
                        (spec > threshold * col_medians), 0, 1)
    else:
        mask = np.where((spec > threshold * row_medians) &
                        (spec > threshold * col_medians), 1, 0)
    return mask


def morphological_filter(mask):
    """ Morphological operation to enhance signal segments. Literature reports at
    least two different methods: 
    - Lasseck 2013: Closing followed by dilation 
    - Sprengler 2017: Opening

    We experimentally developed our own approach: Opening followed by another dilation:
    """
    op = morphology.binary_opening(
        mask, structure=np.ones((4, 4))).astype(np.int)
    dil = morphology.binary_dilation(
        op, structure=np.ones((4, 4))).astype(np.int)
    return dil


def indicator_vector(morph, inv=False):
    """ Takes a binary mask and computes a time scale indicator vector """
    if inv: 
        vec = np.min(morph, axis=0).reshape(1, -1)
    else:
        vec = np.max(morph, axis=0).reshape(1, -1)
    vec = morphology.binary_dilation(
        vec, structure=np.ones((1, 15))).astype(np.int)
    #vec = np.repeat(vec, morph.shape[0], axis=0)
    return vec


def vector_to_timestamps(vec, audio, sr):
    """ Turns an indicator vector into timestamps in seconds """
    # Pad with zeros to ensure that starts and stop at beginning and end are being picked up
    vec = np.pad(vec.ravel(), 1, mode='constant', constant_values=0)
    starts = np.empty(0)
    stops = np.empty(0)
    for i, e in enumerate(vec):
        if e == 1 and vec[i-1] == 0:
            start = i-1                     # Subtract 1 because of padding
            starts = np.append(starts, start)
        if e == 0 and vec[i-1] == 1:
            stop = i-1
            stops = np.append(stops, stop)

    ratio = audio.shape[0] / vec.shape[0]
    timestamps = np.vstack([starts, stops])
    
    # Scale up to original audio length
    timestamps *= ratio
    # Divide by sample rate to get seconds
    timestamps /= sr
    timestamps = np.round(timestamps, 3)

    # Get total duration of signal
    sum_signal = np.sum(timestamps[1] - timestamps[0])
    return json.dumps([tuple(i) for i in timestamps.T]), sum_signal
    
def signal_noise_separation(audio):
    """ Directly returns signal and noise components for a selected raw audio
    vector. Used for precomputing slices when storing timestamps is unnecessary. """
    stft = normalized_stft(audio)
    mask = median_mask(stft, 3)
    morph = morphological_filter(mask)
    vec = indicator_vector(morph)
    ratio = audio.shape[0] // vec.shape[1] #Results in miniscule time dilation of ~0.001 seconds but is safe
    vec_stretched = np.repeat(vec, ratio).astype(bool)

    signal_indeces = np.where(vec_stretched)[0]
    noise_indeces = np.where(~vec_stretched)[0]

    signal = audio[signal_indeces]
    noise = audio[noise_indeces]
    return signal, noise

def signal_timestamps(audio, sr):
    """ Takes audio and sample rate from a bird soundfile and returns the overall 
    duration, the total seconds containing bird vocalizations and a json with start and stop
    markers for these bird vocalizations."""
    stft = normalized_stft(audio)
    mask = median_mask(stft, 3)
    morph = morphological_filter(mask)
    vec = indicator_vector(morph)
    timestamps, sum_signal = vector_to_timestamps(vec, audio, sr)
    duration = audio.shape[0] / sr
    return duration, sum_signal, timestamps
