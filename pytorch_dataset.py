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


train_path = './Urban Sound/train'
labels = pd.read_csv(os.path.join(train_path, 'train.csv'))
file_names = os.listdir(os.path.join(train_path, 'Train'))
#Sort numerically
file_names.sort(key = lambda x: int(x.split('.')[0]))
# Append file path column
labels['path'] = ['./Urban Sound/train/Train/' + x for x in  file_names]


class SoundDataset(Dataset):
    def __init__(self, df):
        
        self.labels = df.Class
        self.ids = df.ID
        self.paths = df.path
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, i):
        path = self.paths[i]
        X = self.get_spectogram(path)
        y = self.labels[i]
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

test = SoundDataset(labels)
y = test[0][0]

def plot(ds, x):
    librosa.display.specshow(ds[x][0], y_axis='log', x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(ds[x][1])
    plt.tight_layout()
    return
    
plot(test, 15)