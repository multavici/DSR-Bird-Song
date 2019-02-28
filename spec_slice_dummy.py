#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:51:45 2019

@author: tim
"""

from Dataset.Preprocessing.utils import load_audio, get_signal, slice_audio
from Spectrogram.spectrograms import stft_s

def prepare_slices(path, timestamps, window, stride):
    audio, sr = load_audio(path)
    signal = get_signal(audio, sr, timestamps)
    audio_slices = slice_audio(signal, window, stride)
    return [stft_s(s) for s in audio_slices]

