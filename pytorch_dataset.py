#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:07 2019

@author: tim
"""

"""
A prototype Dataset Class built around the Urban Sound dataset
tbc
"""

import pandas as pd
import librosa
import librosa.display
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re

#TODO:
# Load h5 here
# Adapt dataset to fit new format


###############################################################################
# Dataset Class
class SoundDataset(Dataset):
    def __init__(self, df):
        
        self.label = df.label
        self.ids = df.ID
        self.paths = df.path
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, i):
        path = self.paths[i]
        X = self.get_spectogram(path)
        y = self.label[i]
        return X, y
    
    def load_audio(self, path):
        raw, sr = librosa.load(path)
        print(librosa.get_duration(raw))
        return raw
    
    def get_spectogram(self, path):
        raw = self.load_audio(path)
        S_full, phase = librosa.magphase(librosa.stft(raw))
        return S_full
    
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

        
        
test = SoundDataset(df)

test.plot(101,102)

s, _ = test[101]
librosa.display.specshow(s, y_axis='log', x_axis='time', sr=sr)
plt.tight_layout()
