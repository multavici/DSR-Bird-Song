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
from .Preprocessing.pre_preprocessing import load_audio, get_signal, slice_spectrogram
from multiprocessing import Process, Queue, Event
import time


# Preloader class outline
class Preloader(Process): # Threading here?
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
    def __init__(self, df, spectogram_func, window, stride, event, queue):
        super(Preloader, self).__init__()

        # DF attributes
        self.path = df.path
        self.label = df.label
        self.total_signal = df.total_signal
        self.timestamps = df.timestamps
        self.sum_total_signal = np.sum(df.total_signal)

        # Spectrogram
        self.window = window
        self.stride = stride
        self.spectrogram = spectogram_func

        # Bucket list
        self.ids = list(np.random.permutation(list(range(len(df)))))   #TODO: Think of more sophisticated sampling strategies
        self.bucket_list = []
        self.get_bucket_list()

        # Bucket as a queue
        self.e = event
        self.q = queue

        # Initial fill
        print('[Preloader] Loading initial slices')
        self.update_bucket()

    def run(self):
        while True:
            event_is_set = self.e.wait()
            if event_is_set:
                print('[Preloader] Refilling bucket...')
                s = time.time()
                #some check here if there is more data available or if last batch will be shorter
                self.get_bucket_list()
                self.update_bucket()
                self.e.clear()
                print(f'[Preloader] Done after {time.time() -s} seconds.')

    def get_bucket_list(self):
        self.bucket_list = self.ids[:6]
        del self.ids[:6]
        return

    def update_bucket(self):
        for idx in self.bucket_list:
            path = self.path[idx]
            timestamps = self.timestamps[idx]
            label = self.label[idx]
            audio, sr = load_audio(path)
            signal = get_signal(audio, sr, timestamps)
            spec = self.spectrogram(signal)
            slices = slice_spectrogram(spec, self.window, self.stride)

            labels = len(slices) * [label]
            for slice_ ,label in zip(slices, labels):
                self.q.put((slice_, label))
        return



# Dataset Class
class SoundDataset(Dataset):
    def __init__(self, df, batchsize, window = 300, stride = 100, spectrogram_func = None, augmentation_func = None):
        """ Initialize with a dataframe containing:
        (path, label, duration, total_signal, timestamps)
        and pass the desired spectral slice length in miliseconds and the overlap
        between successive spectral slices in miliseconds."""

        self.sum_total_signal = np.sum(df.total_signal)
        self.length = int(self.sum_total_signal * 1000 - (window - stride) // stride)

        e = Event()
        self.q = Queue()

        self.augment = augmentation_func

        self.Preloader = Preloader(df, spectrogram_func, window, stride, e, self.q)
        self.Preloader.start()
        #self.Preloader.join()

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
        self.check_bucket()
        X, y = self.q.get()
        return X, y

    def check_bucket(self):
        if self.q.qsize() <= 2*batchsize:
            if not self.Preloader.e.is_set():
                print('\n Running low')
                self.Preloader.e.set()


###############################################################################
"""

# Test Run
import pandas as pd
from spectrograms import stft_s
from torch.utils.data import DataLoader
import time
BATCHSIZE = 10

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
#df.groupby('label').agg({'total_signal' : 'sum'}).plot.bar()

# Instantiate and do training loop

test_ds = SoundDataset(df, spectrogram_func = stft_s)

test_dl = DataLoader(test_ds, batch_size=BATCHSIZE)

for i, batch in enumerate(test_dl):
    print('\n', batch[0].shape, batch[1].shape, test_ds.q.qsize())
    for i in range(5):
        print(f'{i+1}', end = '')
        time.sleep(1)

"""
