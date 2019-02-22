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
from imageio import imwrite

train_path = 'Test Birdsounds'
labels = [name.split('.')[0].replace('%20', '_') for name in os.listdir(train_path) if name[0] != '.']
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

samples = {'label' : [],
           'path'  : []}
c = 1
for path, label in zip(df.path, df.label):
    print(path)
    raw, sr = librosa.load(path)
    a, phase = librosa.magphase(librosa.stft(raw))
    a -= a.min()
    a /= a.max()
    a *= 255
    a = a.astype('uint8') 
    for i in range(0, a.shape[1]-200, 100):
        slice_ = a[:, i:i+200] 
        path = f'spectral_slices/{c}.png'
        samples['label'] += [label]
        samples['path'] += [path]
        imwrite(path, slice_)
        c += 1

sample_data = pd.DataFrame(samples)
sample_data.to_csv('spectral_slices.csv')
