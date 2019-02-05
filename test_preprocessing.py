#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:00:47 2019

@author: tim
"""

import pandas as pd
import librosa
import os
import re

train_path = 'Test Birdsounds'
labels = [name.split('.')[0].replace('%20', ' ') for name in os.listdir(train_path) if name[0] != '.']
labels = [re.sub(r'\d+', '', label) for label in labels]
paths = [os.path.join(train_path, name) for name in os.listdir(train_path) if name[0] != '.']
ids = list(range(len(labels)))
df = pd.DataFrame({'ID' : ids, 'label' : labels, 'path' : paths })

###############################################################################
# Get sound file durations:

durations = []
for path in df.path:
    raw, sr = librosa.load(path)
    dur = librosa.get_duration(raw)
    durations.append(dur)
    
df['duration'] = durations

#short_ones = df[df.duration < 20]

###############################################################################
# Compute spectogram for each sound file and split spectograms into len == 100 segments
# with an overlap of 50 -> store into lobaled dictionary and export as json

samples = {}
i = 1
for path, label in zip(df.path, df.label):
    print(i)
    raw, sr = librosa.load(path)
    a, phase = librosa.magphase(librosa.stft(raw))
    list_ = [a[:, i:i+100] for i in range(0, a.shape[1]-100, 50)]
    label = label.replace(' ', '_')
    if label not in samples.keys():
        samples[label] = list_
    else:
        samples[label] += list_
    i += 1


    
import deepdish as dd
dd.io.save('spectral_slices.h5', samples)

