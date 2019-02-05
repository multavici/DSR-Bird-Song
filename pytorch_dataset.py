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


train_path = 'Test Birdsounds'
labels = [name.split('.')[0] for name in os.listdir(train_path)]
paths = [os.path.join(train_path, name) for name in os.listdir(train_path)]
ids = list(range(4))

df = pd.DataFrame({'ID' : ids, 'label' : labels, 'path' : paths })
"""
labels = pd.read_csv(os.path.join(train_path, 'train.csv'))
file_names = os.listdir(os.path.join(train_path, 'Train'))
#Sort numerically
file_names.sort(key = lambda x: int(x.split('.')[0]))
# Append file path column
labels['path'] = ['./Urban Sound/train/Train/' + x for x in  file_names]
"""

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
        print(sr)
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
        for i in range(start, stop):
            s, _ = self[i]
            ax = plt.subplot(rows, cols, i+1)
            librosa.display.specshow(s, y_axis='log', x_axis='time', sr=sr)
        plt.tight_layout()

        
        
test = SoundDataset(df)

test.plot(0,4)
