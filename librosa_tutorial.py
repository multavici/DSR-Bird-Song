#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:23:36 2019

@author: tim
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

"""
By default, all audio is mixed to mono and resampled to 22050 Hz at load time. 
This behavior can be overridden by supplying additional arguments to librosa.load()
"""
# Raw audio and sampling rate
y, sr = librosa.load('falcon.wav')

# Length in seconds
len_ = len(y)/sr

# time-series harmonic-percussive separation
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# extract the Mel-frequency cepstral coefficients from the raw signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)


# (smoothed) first-order differences among columns of its input
mfcc_delta = librosa.feature.delta(mfcc)

# a chromagram using just the harmonic component
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))


#Plot the spectogram
librosa.display.specshow(S_full, y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()