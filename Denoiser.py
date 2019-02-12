#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:27:26 2019

@author: tim
"""

"""
Following the methodology described in Lasseck 2013 and Sprengler 2017

"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import morphology
# Load audio and compute spectrogram:

y, sr = librosa.load('Test Birdsounds/anser%20brachyrhynchus42.mp3')

spec, phase = librosa.magphase(librosa.stft(y))


def median_mask(spec, threshold, inv = False):
    row_medians = np.expand_dims(np.median(spec, axis = 1), axis = 1)
    col_medians = np.expand_dims(np.median(spec, axis = 0), axis = 0) 
        
    # Broadcasting the median vectors over the spectrogram:
    if inv:
        mask = np.where((spec > threshold * row_medians) & (spec > threshold * col_medians), 0, 1)
    else:
        mask = np.where((spec > threshold * row_medians) & (spec > threshold * col_medians), 1, 0)
    return mask

mask = median_mask(spec, 3)

def morphological_filter(mask):
    """ 
    
    Lasseck 2013: Opening followed by dilation: morphology.binary_dilation(morphology.binary_closing(mask)).astype(np.int)
        
    Sprengler 2017: Closing: morphology.binary_closing(mask)).astype(np.int)
    
    """
    return morphology.binary_opening(mask, structure = np.ones((4,4))).astype(np.int) 

morph = morphological_filter(noise_mask)


def indicator_vector(morph):
    

    
noise_mask = median_mask(spec, 2.5, True)


plt.subplot(311)
plt.imshow(morph)
plt.subplot(312)
plt.imshow(noise_mask)
plt.subplot(313)
plt.imshow(spec)