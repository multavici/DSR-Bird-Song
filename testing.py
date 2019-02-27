#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:19:40 2019

@author: tim
"""


import pandas as pd
from Spectrogram.spectrograms import stft_s
df = pd.read_csv('Testing/test_df.csv')
import time

BATCHSIZE = 64
params = {'batchsize' : BATCHSIZE, 
          'window' : 5000, 
          'stride' : 2000, 
          'spectrogram_func' : stft_s, 
          'augmentation_func' : None}


from Datasets.dynamic_dataset import SoundDataset
from torch.utils.data import DataLoader

ds = SoundDataset(df, **params)

dl = DataLoader(ds, BATCHSIZE)



for batch in dl:
    print(batch[0].shape)
    for i in range(BATCHSIZE):
        time.sleep(1)
        print(i+1)
        
        
ds.Preloader.terminate()


ds.shape
