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
from pandas.api.types import is_numeric_dtype
import os

class SpectralDataset(Dataset):
    """ For fast training of models with precomputed spectrogram slices: """
    def __init__(self, df, input_dir, augmentation_func=None, enhancement_func=None):
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

        self.input_dir = input_dir
        self.augmentation_func = augmentation_func
        self.enhancement_func = enhancement_func

        self.shape = (self[0][0].shape[1], self[0][0].shape[2])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = self.df.path.iloc[i]
        full_path = os.path.join(self.input_dir, path)

        y = self.df.label.iloc[i]
        X = self.unpickle(full_path)
        X -= X.min()
        X /= X.max()
        X = np.expand_dims(X, 0)

        if not self.augmentation_func is None:
            X = self.augmentation_func(X)

        if not self.enhancement_func is None:
            X = self.enhancement_func(X)

        return (X, y)

    def unpickle(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_

class RandomSpectralDataset(SpectralDataset):
    """ Rather than returning a sequential list of files, this dataset can be "blown" up to any
    reasonable size with the slices_per_class parameter. The dataset will then loop through its classes
    with the help of modulo, returning heterogenous batches. The paramete examples_per_batch controls
    how many examples of one class are to be shown in one batch.

    Example: Batchsize 8, nr of classes 12, examples_per_class = 3, slices_per_class 100:
        len = 1200
        first batch:  [0,0,0,1,1,1,2,2]
        second batch: [2,3,3,3,4,4,4,5]
        third batch:  [5,5,6,6,6,7,7,7]
        etc.

    Naturally the ability to blow up the dataset should only be used in combination with
    random augmentations.
    """
    def __init__(self, df, input_dir, slices_per_class=300, examples_per_batch=1, augmentation_func=None, enhancement_func=None):
        """ Initialize with a dataframe containing:
        path for a pickled precomputed spectrogram slice"""
        super(RandomSpectralDataset, self).__init__()

        self.slices_per_class = slices_per_class
        self.examples_per_batch = examples_per_batch
        self.classes = len(set(self.df.label))

    def __len__(self):
        return self.classes * self.slices_per_class

    def __getitem__(self, i):
        y = (min(i, (i // self.examples_per_batch))) % self.classes
        path = self.df[self.df.label == y].sample(n=1).path.values[0]
        full_path = os.path.join(self.input_dir, path)
        X = self.unpickle(full_path)
        X -= X.min()
        X /= X.max()
        X = np.expand_dims(X, 0)

        if not self.augmentation_func is None:
            X = self.augmentation_func(X)

        if not self.enhancement_func is None:
            X = self.enhancement_func(X)

        return (X, y)

    def unpickle(self, path):
        with open(path, 'rb') as f:
            slice_ = pickle.load(f)
        return slice_
