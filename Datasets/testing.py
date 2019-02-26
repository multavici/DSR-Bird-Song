#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:19:40 2019

@author: tim
"""

import pandas as pd
from Spectrogram.spectrograms import stft_s
df = pd.read_csv('Testing/test_df.csv')


params = {'batchsize' : 10, 
          'window' : 1500, 
          'stride' : 500, 
          'spectrogram_func' : stft_s, 
          'augmentation_func' : None}


from Datasets.dynamic_dataset import SoundDataset

ds = SoundDataset(df, **params)

ds.check_stack()


ds.q.qsize()

ds.Preloader.update_bucket()

o2 = ds.q.get()


s = ds.stack

ds.receive_bucket()


ds.Preloader.bucket_list

ds.Preloader.e.set()