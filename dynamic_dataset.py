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
from torch.utils.data import Dataset
import numpy as np
import json


# Preloader class outline
class Preloader(): # Threading here?
    """ always preload a specific number of audiofiles and for each extract
    signal, compute spec, and slice spec. Store in two lists or simply dict
    with label and slices, let getitem take from this one.
    Figure a way to refresh this in the background. 
    And come up with an idea for a sensible dataset length...

    
    This class should be able to take a df from SoundDataset and group it by
    classes as a start. 
    It will then begin filling up a cache of spectrogram slices aiming to always
    ensure equal class representation in its cache. From this cache, SoundDataset
    will grab files when queried. 
    
    
    -> Memory problem: are slices deleted once they were retrieved? Can they be
    re-retrieved? Is it necessary? 
    """
    def __init__(self, df, spectogram_func, window, stride):
        self.sr = 22050
        self.path = df.path
        self.label = df.label
        self.total_signal = df.total_signal
        self.timestamps = df.timestamps
        self.sum_total_signal = np.sum(df.total_signal)
        self.window = window
        self.stride = stride
        
        self.spectrogram = spectogram_func
        
        self.ids = list(np.random.permutation(list(range(len(df)))))   #TODO: Think of more sophisticated sampling strategies
        
        self.bucket_list = []
        self.get_bucket_list()
        
        self.bucket = []
        self.update_bucket()
            
    def query(self):
        q = self.bucket.pop(0)   # Always first item
        
        #Perform check on bucket size
        if len(self.bucket) < BATCHSIZE:
            print('Refilling bucket...')
            #some check here if there is more data available or if last batch will be shorter
            self.get_bucket_list()
            self.update_bucket()     
        return q
 
    def get_bucket_list(self):
        self.bucket_list = self.ids[:4]
        del self.ids[:4]
        return
    
    def update_bucket(self):
        for idx in self.bucket_list:
            path = self.path[idx]
            timestamps = self.timestamps[idx]
            label = self.label[idx]
            
            audio = self.load_audio(path)
            signal = self.get_signal(audio, timestamps)
            spec = self.spectrogram(signal)
            slices = self.slice_spectrogram(spec)
            
            labels = len(slices) * [label]
            self.bucket += [(slice_, label) for slice_ ,label in zip(slices, labels)]
        return   
    
    def load_audio(self, path):
        """ Audio i/o """
        audio, sr = librosa.load(path)
        #assert sr == self.sr
        return audio
    
    def get_signal(self, audio, timestamps):
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
        window = 200#int(self.window / 1000 * self.sr)
        stride = 100#int(self.stride / 1000 * self.sr)
        return [spec[:, i:i+window] for i in range(0, spec.shape[1]-window, stride)]



# Dataset Class
class SoundDataset(Dataset):
    def __init__(self, df, window = 300, stride = 100, spectrogram_func = None, augmentation_func = None):
        """ Initialize with a dataframe containing: 
        (path, label, duration, total_signal, timestamps) 
        and pass the desired spectral slice length in miliseconds and the overlap 
        between successive spectral slices in miliseconds."""
        
        self.sum_total_signal = np.sum(df.total_signal)
        self.length = int(self.sum_total_signal * 1000 - (window - stride) // stride)
        
        self.augment = augmentation_func
        
        self.Preloader = Preloader(df, spectrogram_func, window, stride)
        
    def __len__(self):
        """ The length of the dataset is not the number of audio 
        files but the maximum number of bird vocalization slices that could be
        extracted from the sum total of vocalization parts given a slice duration 
        and an offset.
        ."""
        return 1000  #self.length    #TODO: Give an actual safe estimate here.
        
    def __getitem__(self, i):
        """ Indices become meaningless here... The preloader returns items until 
        it runs out. """
        X, y = self.Preloader.query()
        return X, y
        
    
###############################################################################
# Test Run
from spectrograms import stft_s
from torch.utils.data import DataLoader

BATCHSIZE = 64

df = pd.read_csv('test_df.csv')

def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col

df.label = label_encoder(df.label)    



# Instantiate and do training loop

test_ds = SoundDataset(df, spectrogram_func = stft_s)

test_dl = DataLoader(test_ds, batch_size=BATCHSIZE)

for i, batch in enumerate(test_dl):
    print(batch[0].shape, batch[1])



