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

from torch.utils.data import Dataset
import numpy as np
from .Preprocessing.pre_preprocessing import load_audio, get_signal
from multiprocessing import Process, Queue, Event, active_children
from multiprocessing.pool import ThreadPool
import time


class Preloader(Process):
    def __init__(self, event, queue, task_queue):
        super(Preloader, self).__init__()

        # Bucket list -> list of files that its supposed to get - updated by Dataset
        self.bucket_list = []

        self.e = event
        self.q = queue
        self.t = task_queue
        
    def run(self):
        while True:
            event_is_set = self.e.wait()
            if event_is_set:
                print('[Preloader] Refilling bucket...')
                bucket_list = self.t.get()
                self.bucket_list = bucket_list
                self.update_bucket()
                self.e.clear()

                
    def update_bucket(self):
        print('check')
        print(self.bucket_list)
        pool = ThreadPool(4)
        output = pool.map(self.preload, list(range(len(self.bucket_list))))
        print('check2')
        self.q.put(output)
    
    def preload(self, i):
        p, l, t = self.bucket_list[i]
        print(p, l)
        audio, sr = load_audio(p)
        signal = get_signal(audio, sr, t)
        return (l, list(signal))
        


class SoundDataset(Dataset):
    def __init__(self, df, **kwargs):
        """ Initialize with a dataframe containing:
        (path, label, duration, total_signal, timestamps)
        kwargs: batchsize = 10, window = 1500, stride = 500, spectrogram_func = None, augmentation_func = None"""
        
        self.df = df
        self.df['loaded'] = 0
        self.sr = 22050
        
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # Window and stride in samples:
        self.window = self.window/1000*self.sr
        self.stride = self.stride/1000*self.sr


        # Stack - a list of continuous audio signal for each class
        self.stack = {label:[] for label in set(self.df.label)}
        self.classes = len(set(self.df.label))
        
        # Instantiate Preloader:
        e = Event()
        self.q = Queue()
        self.t = Queue()
        self.Preloader = Preloader(e, self.q, self.t)
        self.Preloader.start()



    def __len__(self):
        """ The length of the dataset is the (expected) maximum number of bird vocalization slices that could be
        extracted from the sum total of vocalization parts given a slicing window
        and stride."""
        return 100  #self.length    #TODO: Give an actual safe estimate here.

    def __getitem__(self, i):
        """ Indices become meaningless here... The preloader returns items until
        it runs out. """
        self.check_stack()
        y = i % self.classes # loop through classes 
        X = self.stack[y][:int(self.window)]     #Extract audio corresponding to window length
        del self.stack[y][:int(self.stride/1000*self.sr)]     #Delete according to stride length
        
        return X, y

    def check_stack(self):
        # Check for short lists in stack
        bucket_list = []
        for k,v in self.stack.items():
            if len(v) < 2* self.window:
                bucket_list.append(self.make_request(k))

        # If preloader is not already working:
        if len(bucket_list) > 0 and not self.Preloader.e.is_set():
            
            self.t.put(bucket_list)   #update the bucket list
            self.Preloader.e.set()

                
    def receive_bucket(self):
        if not self.q.empty():
            bucket = self.q.get()
            for sourcefile in bucket:
                # Figure out how data is stored by Preloader
                l = sourcefile[0]
                self.stack[l].append(list(sourcefile[1]))
                
    
    def make_request(self, k):
        # search through the df and pick files for loading
        sample = self.df[self.df.label == k].sample(n=1)
        path = sample.path.values[0]
        label = sample.label.values[0]
        timestamps = sample.timestamps.values[0]
        return (path, label, timestamps)
        


###############################################################################
"""
# Test Run
import pandas as pd
from Spectrogram.spectrograms import stft_s
from torch.utils.data import DataLoader
BATCHSIZE = 10

df = pd.read_csv('Testing/test_df.csv')

def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col


df.label = label_encoder(df.label)
#df.groupby('label').agg({'total_signal' : 'sum'}).plot.bar()

# Instantiate and do training loop

test_ds = SoundDataset(df, BATCHSIZE, spectrogram_func = stft_s)

test_dl = DataLoader(test_ds, batch_size=BATCHSIZE)

for i, batch in enumerate(test_dl):
    print('\n', batch[0].shape, batch[1].shape, test_ds.q.qsize())
    for i in range(5):
        print(f'{i+1}', end = '')
        time.sleep(1)

"""
