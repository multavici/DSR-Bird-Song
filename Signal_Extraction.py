#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:27:26 2019

@author: tim
"""

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
    stft = np.abs(librosa.stft(audio, n_fft = 2048)) #2048, win_length = 512))
    return stft / stft.max()

def median_mask(spec, threshold, inv = False):
    """ Returns a binary mask for values that are above a threshold times the row
    and column medians. Inverting the mask is possible. """
    row_medians = np.expand_dims(np.median(spec, axis = 1), axis = 1)
    col_medians = np.expand_dims(np.median(spec, axis = 0), axis = 0)  
    # Broadcasting the median vectors over the spectrogram:
    if inv:
        mask = np.where((spec > threshold * row_medians) & (spec > threshold * col_medians), 0, 1)
    else:
        mask = np.where((spec > threshold * row_medians) & (spec > threshold * col_medians), 1, 0)
    return mask

def morphological_filter(mask):
    """ Morphological operation to enhance signal segments. Literature reports at
    least two different methods: 
    - Lasseck 2013: Closing followed by dilation: morphology.binary_dilation(morphology.binary_closing(mask)).astype(np.int)    
    - Sprengler 2017: Opening: morphology.binary_closing(mask)).astype(np.int)
    
    We experimentally developed our own approach: Opening followed by another dilation:
    """
    op = morphology.binary_opening(mask, structure = np.ones((4,4))).astype(np.int) 
    dil = morphology.binary_dilation(op, structure = np.ones((4,4))).astype(np.int) 
    return dil

def indicator_vector(morph):
    """ Takes a binary mask and computes a time scale indicator vector """
    vec = np.max(morph, axis = 0).reshape(1, -1)
    vec = morphology.binary_dilation(vec, structure = np.ones((1,15))).astype(np.int)
    #vec = np.repeat(vec, morph.shape[0], axis=0)
    return vec
    
def vector_to_timestamps(vec, audio, sr):
    """
    """
    #Pad with zeros to ensure that starts and stop at beginning and end are being picked up
    vec = np.pad(vec.ravel(), 1, mode = 'constant', constant_values = 0)
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
    
    try:
        timestamps = np.vstack([starts, stops])
    except: 
        print(f'Nr. of starts and stops doesnt match for file {path}') # Technically impossible but I'm superstitiuous
    
    #Scale up to original audio length
    timestamps *= ratio
    #Divide by sample rate to get seconds
    timestamps /= sr
    timestamps = np.round(timestamps, 3)
    
    #Get total duration of signal    
    sum_signal = np.sum(timestamps[1] - timestamps[0])
    return json.dumps([tuple(i) for i in timestamps.T]), sum_signal


def signal_timestamps(path):
    """ Takes an audio path to a bird soundfile and returns the overall duration,
    the total seconds containing bird vocalizations and a json with start and stop
    markers for these bird vocalizations."""
    audio, sr = librosa.load(path)
    stft = normalized_stft(audio)
    mask = median_mask(stft, 3)
    morph = morphological_filter(mask)
    vec = indicator_vector(morph)
    timestamps, sum_signal = vector_to_timestamps(vec, audio, sr)
    duration = audio.shape[0] / sr
    return duration, sum_signal, timestamps
    


