#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:20:34 2019

@author: tim
"""
import numpy as np
import librosa
from .Chirplet import FCT

### 1. A SIMPLE STFT
def stft_s(audio):
    return np.abs(librosa.stft(audio, center=False))

##### 2. Mel  spectorgram
def mel_s(audio):
    sr = 22050
    return librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)

##### 3. chirp
def chirp_s(audio):
    return FCT().compute(audio)


