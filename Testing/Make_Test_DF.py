#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:10:43 2019

@author: tim
"""


"""
Just a test dataframe used during development of dynamic dataset

"""

from Signal_Extraction import signal_timestamps
import time
import re

# Making a test df
DIR = 'Test Birdsounds'

p = []
l = []
d = []
s = []
t = []
loadt = [] # Just for testing speed
for path in [path for path in os.listdir(DIR) if path.endswith('.mp3')]:
    start = time.time()
    label = re.sub('\d+', '', path.split('.')[0].replace('%20', '_'))    
    path = os.path.join(DIR, path)
    dur, sig, tim = signal_timestamps(path)
    p.append(path)
    l.append(label)
    d.append(dur)
    s.append(sig)
    t.append(tim)
    loadt.append(time.time() - start)
    
test_df = pd.DataFrame({'path' : p, 'label' : l, 'duration': d, 'total_signal': s, 'timestamps' : t, 'load_time' : loadt})
test_df['load_time/s'] = test_df.load_time / test_df.duration # speed in relation to audio duration


test_df.to_csv('test_df.csv')

# How long would it approximately take to process 100k audio files?
np.mean(test_df.load_time) * 100.000 / 60  





######################################################
def plot_example(audio, signal):
    plt.subplot(221)
    librosa.display.waveplot(audio, x_axis='time', sr=sr)
    plt.subplot(223)
    librosa.display.specshow(np.log(librosa.stft(audio)), y_axis='log', x_axis='time', sr=sr)
    plt.subplot(222)
    librosa.display.waveplot(signals, x_axis='time', sr=sr)
    plt.subplot(224)
    librosa.display.specshow(np.log(librosa.stft(signals)), y_axis='log', x_axis='time', sr=sr)
    return