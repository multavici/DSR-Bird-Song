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
"""

from torch.utils.data import Dataset
from .Preprocessing.pre_preprocessing import load_audio, get_signal
from multiprocessing.pool import ThreadPool
import numpy as np


class SoundDataset(Dataset):
    def __init__(self, df, **kwargs):
        """ Initialize with a dataframe containing:
        (path, label, duration, total_signal, timestamps)
        kwargs: batchsize = 10, window = 1500, stride = 500, spectrogram_func = None, augmentation_func = None"""
        
        self.df = df
        self.df.loc['loaded'] = 0
        self.sr = 22050
        
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # Window and stride in samples:
        self.window = int(self.window/1000*self.sr)
        self.stride = int(self.stride/1000*self.sr)


        # Stack - a list of continuous audio signal for each class
        self.stack = {label:[] for label in set(self.df.label)}
        self.classes = len(set(self.df.label))
        
        self.length = self.compute_length()

    def __len__(self):
        """ The length of the dataset is the (expected) maximum number of bird 
        vocalization slices that could be extracted from the sum total of 
        vocalization parts given a slicing window and stride."""
        return self.length    #TODO: Give an actual safe estimate here.

    def __getitem__(self, i):
        """ Indices become meaningless here, they only ensure that subsequent 
        samples are of different classes. """
        self.check_stack()
        y = i % self.classes # loop through classes 
        X = self.stack[y][:self.window]     #Extract audio corresponding to window length
        self.stack[y] = np.delete(self.stack[y], np.s_[:self.stride])     #Delete according to stride length
        
        X = self.spectrogram_func(X)
        #TODO: Process to check for which files to augment:
        """ 
        if self.augmentation_func not None:
            X = self.augmentation_func(X)
        """
        return X, y

    def check_stack(self):
        """ Check for classes in the stack that do not have enough audio left 
        on storage to serve at least two more times. If such exist, store a
        request for them in the bucket list. Since the preloader only becomes
        active when at least 4 such files exist, also raise an alarm if one class
        in the stack is almost depleted and preload right away. """
        bucket_list = []
        alarm = False
        for k,v in self.stack.items():
            remaining_serves = ((len(v) - self.window) / self.stride + 1 )
            if remaining_serves < 3:
                bucket_list.append(self.make_request(k))
            if remaining_serves < 2:
                alarm = True

        # If any classes are running low:
        if len(bucket_list) >= 4 or alarm:
            print(f'Found {len(bucket_list)} files to preload')
            # Start loading
            new_samples = self.preload(bucket_list)
            self.receive_request(new_samples)
            
    def preload(self, bucket_list):
        """ Initiates a number of threads (max 8, otherwise nr. of files in
        bucket_list)
        """
        def _preload(i):
            p, l, t = bucket_list[i]
            audio, sr = load_audio(p)
            signal = get_signal(audio, sr, t)
            return (l, signal)
        nr_threads = min([8, len(bucket_list)])
        print(f'Initiating {nr_threads} threads to preload files')
        pool = ThreadPool(nr_threads)
        output = pool.map(_preload, list(range(len(bucket_list))))
        return output 
            
    def make_request(self, k):
        """ pick a random file from a class that is running low in the stack and
        return path, label, and timestamps for that file. """
        sample = self.df[self.df.label == k].sample(n=1) 
        path = sample.path.values[0]
        label = sample.label.values[0]
        timestamps = sample.timestamps.values[0]
        return (path, label, timestamps)
    
    def receive_request(self, new_samples):
        """ Takes a newly loaded of label, audio tuples and sorts them into the
        stack. """
        print(f'Done preloading, sorting into stack')
        for sample in new_samples:
            label = sample[0]
            self.stack[label] = np.append(self.stack[label], (sample[1]))
        
    def compute_length(self):
        sum_total_signal = sum(self.df.total_signal) * 22050
        max_samples = ((sum_total_signal - self.window) // self.stride) + 1
        return int(max_samples)
        
