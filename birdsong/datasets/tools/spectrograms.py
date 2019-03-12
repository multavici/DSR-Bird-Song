#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:20:34 2019

@author: tim
"""
import numpy as np
import librosa
from .dependencies.chirplet import FCT

### 1. A SIMPLE STFT
def stft_s(audio):
    """Short-Time Fourier Transformation"""
    return np.abs(librosa.stft(audio, center=False))

##### 2. Mel  spectorgram
def mel_s(audio, n_mels = 128, fmin=0, fmax=12000):
    """Mel spectrogram"""
    return librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=n_mels, fmin=fmin, fmax=fmax)

##### 3. chirp
def chirp_s(audio, 
            duration_longest_chirplet=1,
            num_octaves=8,
            num_chirps_by_octave=20,
            polynome_degree=1,
            end_smoothing=0.001,
            sample_rate=22050):
    """Chirplet Spectrogram"""
    return FCT(duration_longest_chirplet,
               num_octaves,
               num_chirps_by_octave,
               polynome_degree,
               end_smoothing,
               sample_rate).compute(audio)