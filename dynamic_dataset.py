#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:07 2019

@author: tim
"""

"""
This is dataset class prototype that is initiated from a df containing 5 columns:
path, label, duration, total signal duration, signal timestamps. Those correspond
to a soundfile with birdsong, the foreground species, the total length of the file,
the total length of segments identified as containing bird vocalizations, and 
timestamps of where those vocalizations occur.

The dataset class is supposed to dynamically:
    - load an audio file
    - slice and concatenate bird vocalization segments
    - compute a spectrogram of desired type and parameters for these segments
    - slice segments into specified lengths
    - potentially augment slices
    - collate a random selection of slices into a batch
    
Remaining questions are:
    - How to best handle the one-to-many relationship of audio file to spectrogram slices
    - How to ensure equal class representation while making use of all data
    - How to design loading computationally efficient so that it can be performed
    parallel during training
    
"""

import pandas as pd
import librosa
import librosa.display
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import json


# Dataset Class
class SoundDataset(Dataset):
    def __init__(self, df, slice_dur = 300, stride = 100, spectrogram_func = None, augmentation_func = None):
        """ Initialize with a dataframe containing: 
        (path, label, duration, total_signal, timestamps) 
        and pass the desired spectral slice length in miliseconds and the overlap 
        between successive spectral slices in miliseconds."""
        
        self.slice_dur = slice_dur
        self.stride = stride
        self.path = df.path
        self.label = df.label
        self.total_signal = df.total_signal
        self.timestamps = df.timestamps
        self.sum_total_signal = np.sum(df.total_signal)
        self.length = int(self.sum_total_signal * 1000 - (slice_dur - stride) // stride)
        self.sr = 22050
        self.spectrogram = spectrogram_func
        self.augment = augmentation_func
        
    def __len__(self):
        """ The length of the dataset is not the number of audio 
        files but the maximum number of bird vocalization slices that could be
        extracted from the sum total of vocalization parts given a slice duration 
        and an offset. 
        Calculated on initialization to save computation."""
        return self.length
        
    def __getitem__(self, i):
        
        pass
    
    def preload_batch():
        """ always preload a specific number of audiofiles and for each extract
        signal, compute spec, and slice spec. Store in two lists or simply dict
        with label and slices, let getitem take from this one.
        Figure a way to refresh this in the background. 
        And come up with an idea for a sensible dataset length...
        
        
        """
        
        
        return
        
    def get_signal_spec_slices(self, idx):
        signal, label = self.get_signal(idx)
        slices = self.slice_spectrogram(self.spectrogram(signal))
        return slices
    
    def get_signal(self, idx):
        path = self.path[idx]
        label = self.label[idx]
        timestamps = self.timestamps[idx]
        
        audio = self.load_audio(path)
        signal = self.concat_signal(audio, timestamps)
        return signal, label
    
    def load_audio(self, path):
        """ Audio i/o """
        audio, sr = librosa.load(path)
        assert sr == self.sr
        return audio
    
    def concat_signal(self, audio, timestamps):
        """ Extract and concatenate bird vocalizations at timesteps from audio""" 
        # Convert timestamps from seconds to sample indeces
        timestamps = np.round(np.array(json.loads(timestamps)) * self.sr).astype(np.int)
        r = np.arange(audio.shape[0])
        mask = (timestamps[:,0][:,None] <= r) & (timestamps[:,1][:,None] >= r)
        # Signal as concatenation of all masked sections
        signal = audio.reshape(1, -1).repeat(mask.shape[0], axis = 0)[mask] 
        return signal
    
    def slice_spectrogram(self, spec):
        # Depending on spec properties, dim1 can vary!
        window = 200#int(self.slice_dur / 1000 * self.sr)
        stride = 100#int(self.stride / 1000 * self.sr)
        return [spec[:, i:i+window] for i in range(0, spec.shape[1]-window, stride)]
    
    def collate_func(self, batch):
        # This is what handles batch loading!
        pass
        """
        images = [b[0][0] for b in batch]
        bbox = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        
        encoded = [self.encoder.encode(bb, l, torch.Tensor([256, 256])) for bb, l in zip(bbox, labels)]
        loc_target = [l[0] for l in encoded]
        cls_target = [l[1] for l in encoded]
    
        return torch.stack(images) / 255, torch.stack(loc_target), torch.stack(cls_target)
        """
        
    def plot(self, start, stop):
        sr = 22050
        l = stop - start
        cols = 4
        rows = l // cols + 1
        for i in range(0, l):
            s, _ = self[i]
            ax = plt.subplot(rows, cols, i+1)
            librosa.display.specshow(s, y_axis='log', x_axis='time', sr=sr)
        plt.tight_layout()

###############################################################################
""" Test data

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
"""
###############################################################################
# Instantiating dataset class

df = pd.read_csv('test_df.csv')
test = SoundDataset(df, spectrogram_func = librosa.stft)



t = test.get_signal_spec_slices(3)








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