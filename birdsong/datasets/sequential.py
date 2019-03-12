#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:07 2019

@author: tim
"""
import pickle
import numpy as np
from .tools.io import load_audio, get_signal
from .tools.encoding import LabelEncoder
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    """ For fast training of models with precomputed spectrogram slices: """
    def __init__(self, df, augmentation_func=None, enhancement_func=None):
        """ Initialize with a dataframe containing:
        path for a pickled precomputed spectrogram slice"""
        
        self.df = df
        # Check if labels already encoded and do so if not
        if not is_numeric_dtype(self.df.label):
            self.encoder = LabelEncoder(self.df.label)
            self.df.label = self.encoder.encode()
        else:
            print('Labels look like they have been encoded already, \
            you have to take care of decoding yourself.')

        self.augmentation_func = augmentation_func
        self.enhancement_func = enhancement_func

        self.class_balances = self.df.groupby('label').path.count()
        

        self.shape = (self[0][0].shape[1], self[0][0].shape[2])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = self.df.path.iloc[i]
        y = self.df.label.iloc[i]
        X = self.unpickle(path)
        X -= X.min()
        X /= X.max()
        X = np.expand_dims(X, 0)
        
        if not self.augmentation_func is None:
            X = self.augmentation_func(X)
            
        if not self.enhancement_func is None:
            X = self.augmentation_func(X)
        
        return (X, y)

    def unpickle(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_
