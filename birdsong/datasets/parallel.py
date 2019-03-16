#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:07 2019

@author: tim
"""

"""
This is a PyTorch compatible Dataset class that is initiated from a Dataframe
containing 5 columns: path, label, duration, total signal duration, signal timestamps.
Those correspond to a soundfile with birdsong, the foreground species, the total
length of the file, the total length of segments identified as containing bird
vocalizations, and timestamps of where those vocalizations occur.

The Dataset is combined with a Preloader process which runs in the background
and prepares a new batch of data during each training period.
"""

from torch.utils.data import Dataset
import numpy as np
from .tools.io import load_audio, get_signal
from .tools.encoding import LabelEncoder
from multiprocessing import Process, Queue, Event
from multiprocessing.pool import ThreadPool
from pandas.api.types import is_numeric_dtype
import os
import pickle
import h5py

class Preloader(Process):
    """ A subprocess running in the background
    """
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
                #print('[Preloader] Refilling bucket...')
                self.bucket_list = self.t.get()
                self.preload_batch()
                self.e.clear()
                #print('[Preloader] Done.')

    def preload_batch(self):
        pool = ThreadPool(min(8, len(self.bucket_list)))
        output = pool.map(self._preload, self.bucket_list)
        self.q.put(output)

    def _preload(self, b):
        p, l, t, a = b
        audio, sr = load_audio(p)
        signal = get_signal(audio, sr, t)
        try:
            assert len(signal) >= a
        except AssertionError:
            print(f'Issue with file {p}, supposed to be at least {a} long, but only {len(signal)}')
        return (l, signal.tolist())

class SoundDataset(Dataset):
    def __init__(self, df, **kwargs):
        """ Initialize with a dataframe containing:
        (path, label, duration, total_signal, timestamps)
        kwargs: input_dir = path, batchsize = 10, window = 1500, stride = 500,
        spectrogram_func = None, augmentation_func = None"""
        for k,v in kwargs.items():
            setattr(self, k, v)

        # Ignore recordings with less signal than window size
        self.df = df[df.total_signal >= self.window / 1000]
        self.sr = 22050

        # Check if labels already encoded and do so if not
        if not is_numeric_dtype(self.df.label):
            self.encoder = LabelEncoder(self.df.label)
            self.df.label = self.encoder.encode()
        else:
            print('Labels look like they have been encoded already, \
            you have to take care of decoding yourself.')

        # Window and stride in samples:
        self.window = int(self.window/1000*self.sr)
        self.stride = int(self.stride/1000*self.sr)

        # Stack - a list of continuous audio signal for each class
        self.stack = {label:[] for label in set(self.df.label)}
        self.classes = len(set(self.df.label))

        # Instantiate Preloader:
        e = Event()
        self.q = Queue()
        self.t = Queue()
        self.Preloader = Preloader(e, self.q, self.t)
        self.Preloader.start()

        # Compute total of available slices
        self.length = self.compute_length()

        # Compute output size
        self.shape = self.compute_shape()

        # For troubleshooting the preloader
        self.log = {'sent' : [],
                    'received' : [],
                    'inventory' : []}

        # Prepare the first batch:
        print('Preloading first batch... this might take a moment.')
        self.request_batch(0, -1)

    def __len__(self):
        """ The length of the dataset is the (expected) maximum number of bird vocalization slices that could be
        extracted from the sum total of vocalization parts given a slicing window
        and stride. Calculated by self.compute_length()"""
        return self.length

    def __getitem__(self, i):
        """ Indices loop over available classes to return heterogeneous batches
        of uniformly random sampled audio. Preloading is triggered and the end
        of each batch and supposed to run during training time. At the beginning
        of a new batch, the preloaded audio is received from a queue - if it is
        not ready yet the main process will wait. """
        # Subsequent indices loop through the classes:
        y = i % self.classes

        # If were at the beginning of one batch, update stack with Preloader's work
        if i % self.batchsize == 0:
            self.log['inventory'].append(self.inventory(i))
            self.receive_request(i)

        # Get actual sample and compute spectrogram:
        audio = self.retrieve_sample(y)
        X = self.spectrogram_func(audio)

        # Normlize:
        X -= X.min()
        X /= X.max()
        X = np.expand_dims(X, 0)

        #TODO: Process to check for which files to augment:
        """
        if self.augmentation_func not None:
            X = self.augmentation_func(X)
        """

        # If were at the end of one batch, request next:
        if (i + 1) % self.batchsize == 0:
            self.log['inventory'].append(self.inventory(i))
            self.request_batch(y+1, i)

        # If were at the end of the last batch request starting with class 0
        if i == self.length-1:
            self.log['inventory'].append(self.inventory(i))
            self.request_batch(0, i)

        return X, y

    def retrieve_sample(self, k):
        """ For class k extract audio corresponding to window length from stack
        and delete audio corresponding to stride length"""
        X = self.stack[k][:self.window]
        try:
            assert len(X) == self.window
        except AssertionError:
            import pdb; pdb.set_trace()

        self.stack[k] = np.delete(self.stack[k], np.s_[:self.stride])
        return X

    def receive_request(self, i):
        """ Check if Preloader has already filled the Queue and if so, sort data
        into stack."""
        new_samples = []
        if self.made_request:
            new_samples = self.q.get()
            #print('Queue ready - updating stack.')

            for sample in new_samples:
                label = sample[0]
                self.stack[label] = np.append(self.stack[label], (sample[1]))
            self.made_request = False

        #LOG
        r = {'R':i}
        req = {k: 0 for  k in set([s[0] for s in new_samples])  }
        for s in new_samples:
            req[s[0]] += len(s[1])
        self.log['received'].append({**r, **req})

    def request_batch(self, y, i):
        """ At a given y in classes, look ahead how many times each class k will
        have to be served for the next batch. Request a new file for each where
        only one serving remains.
        If requests are necessary, send them to Preloader.
        """
        samples_needed = self.compute_need(y)
        bucket_list = []
        for k, s in samples_needed.items():
            required_audio = self.check_stack(k, s)
            if required_audio > 0:
                requests = self.make_request(k, required_audio)
                #print(f'Need {required_audio} for class {k}, loading {len(requests)} file(s)')
                for request in requests:
                    bucket_list.append(request)

        if len(bucket_list) > 0:
            #print('Making request')
            self.t.put(bucket_list)
            self.Preloader.e.set()
            self.made_request = True

        #LOG
        r = {'R':i}
        req = {'path':[],
               'label':[],
               'timestamps':[],
               'expected': []}
        for b in bucket_list:
            req['path'].append(b[0])
            req['label'].append(b[1])
            req['timestamps'].append(b[2])
            req['expected'].append(b[3])
        self.log['sent'].append({**r, **req})

    def compute_need(self, y):
        """ Given a current class label, calculate based on batch size how many times
        which classes need to be served in the next batch """
        next_batch = range(y, y + self.batchsize)
        samples_needed = {}
        for i in next_batch:
            k = i % self.classes
            if k in samples_needed.keys():
                samples_needed[k] += 1
            else:
                samples_needed[k] = 1
        return samples_needed

    def check_stack(self, k, s):
        """ Return true if the audio on stack for class k does only suffice for
        a more serves, false if otherwise """
        # Safety buffer added #TODO: find out why this helps
        required_audio = self.window + ((s-1) * self.stride) + 20000
        remaining_audio = len(self.stack[k])
        if remaining_audio < required_audio:
            return required_audio - remaining_audio
        else:
            return 0

    def make_request(self, k, a):
        """ For class k sample a random corresponding sound file and return a
        tuple of path, label, timestamps required by the preloader. """
        request = []
        audio_to_preload = 0
        while audio_to_preload < a:
            sample = self.df[self.df.label == k].sample(n=1)
            path = os.path.join(self.input_dir, sample.path.values[0])
            label = sample.label.values[0]
            timestamps = sample.timestamps.values[0]
            audio_samples = int(sample.total_signal.values[0] * self.sr * 0.9)
            request.append((path, label, timestamps, audio_samples))
            audio_to_preload += audio_samples
        return request

    def inventory(self, i):
        """ Return a dictionary containing the length of audio currently in
        stack for each class. """
        r = {'R':i}
        inv = {k: len(v) for k,v in self.stack.items()}
        return {**r, **inv}


    def compute_length(self):
        """ Provide an estimate of how many iterations of random, uniformly
        sampled audio are needed to have seen each file approximately once. """
        sum_total_signal = sum(self.df.total_signal) * self.sr
        max_samples = ((sum_total_signal - self.window) // self.stride) + 1
        return int(max_samples)

    def compute_shape(self):
        """ Given the possibility of different window sizes and spectrogram functions,
        models need to be initialized with the expected image size. This function
        computes that size and makes it available as the .shape of the Dataset."""
        dummy = np.random.randn(self.window)
        spec = self.spectrogram_func(dummy)
        return spec.shape


################################################################################
class SlicePreloader(Preloader):
    """ A subprocess running in the background, child of audio preloader,
    unpickles the next batch of slices.
    """
    def __init__(self, event, queue, task_queue):
        super(SlicePreloader, self).__init__(event, queue, task_queue)

    def _preload(self, b):
        path, label = b
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                slice_ = pickle.load(f)
        if path.endswith('.h5'):
            h5f = h5py.File(path, 'r')
            slice_ = h5f['sound'][:]
            h5f.close()
        return (label, slice_)

class RandomSpectralDataset(Dataset):
    """ Same concept as the above only that in this case we are preloading a batch
    of precomputed spectrograms, either as pickles or h5. """
    
    def __init__(self, df, **kwargs):
        """ Initialize with a dataframe containing:
        (path, label)
        kwargs: batchsize=64, input_dir='', 
        slices_per_class=300, examples_per_batch=1, augmentation_func=None, 
        enhancement_func=None """
        for k,v in kwargs.items():
            setattr(self, k, v)

        # Ignore recordings with less signal than window size
        self.df = df

        # Check if labels already encoded and do so if not
        if not is_numeric_dtype(self.df.label):
            self.encoder = LabelEncoder(self.df.label)
            self.df.label = self.encoder.encode()
        else:
            print('Labels look like they have been encoded already, \
            you have to take care of decoding yourself.')

        # Stack - a list of continuous audio signal for each class
        self.stack = {label:[] for label in set(self.df.label)}
        self.classes = len(set(self.df.label))

        # Instantiate Preloader:
        e = Event()
        self.q = Queue()
        self.t = Queue()
        self.Preloader = SlicePreloader(e, self.q, self.t)
        self.Preloader.start()

        # Compute output size
        self.shape = (256, 216)
        self.classes = len(set(self.df.label))

        # Prepare the first batch:
        print('Preloading first batch... this might take a moment.')
        self.request_batch(0, 0)

    def __len__(self):
        """ This dataset is capable of automatically up- or downsampling, thus
        the length corresponds to the following: """
        return self.classes * self.slices_per_class

    def __getitem__(self, i):
        """ Indices loop over available classes to return heterogeneous batches
        of uniformly random sampled audio. Preloading is triggered and the end
        of each batch and supposed to run during training time. At the beginning
        of a new batch, the preloaded slices are received from a queue - if they
        are not ready yet the main process will wait. """
        # Subsequent indices loop through the classes:
        #import pdb; pdb.set_trace()
        y = (min(i, (i // self.examples_per_batch))) % self.classes

        # If were at the beginning of one batch, update stack with Preloader's work
        if i % self.batchsize == 0:
            self.receive_request(i)

        # Get spectrogram slice
        X = self.retrieve_sample(y)

        # Normlize:
        X -= X.min()
        X /= X.max()
        X = np.expand_dims(X, 0)

        if not self.augmentation_func is None:
            X = self.augmentation_func(X)

        if not self.enhancement_func is None:
            X = self.enhancement_func(X)

        # If were at the end of one batch, request next:
        if (i + 1) % self.batchsize == 0:
            self.request_batch(y+1, i)

        # If were at the end of the last batch request starting with class 0
        if i == len(self)-1:
            self.request_batch(0, i)

        return X, y

    def retrieve_sample(self, k):
        """ For class k extract audio corresponding to window length from stack
        and delete audio corresponding to stride length"""
        X = self.stack[k].pop(0)
        return X

    def receive_request(self, i):
        """ Check if Preloader has already filled the Queue and if so, sort data
        into stack."""
        new_samples = []
        if self.made_request:
            new_samples = self.q.get()
            #print('Queue ready - updating stack.')

            for sample in new_samples:
                label = sample[0]
                self.stack[label].append(sample[1])
            self.made_request = False
            
    def request_batch(self, y, i):
        """ At a given y in classes, look ahead how many times each class k will
        have to be served for the next batch. Request a new file for each where
        only one serving remains.
        If requests are necessary, send them to Preloader.
        """
        samples_needed = self.compute_need(i)
        bucket_list = []
        for k, s in samples_needed.items():
            required_samples = self.check_stack(k, s)
            if required_samples > 0:
                requests = self.make_request(k, required_samples)
                #print(f'Need {required_audio} for class {k}, loading {len(requests)} file(s)')
                for request in requests:
                    bucket_list.append(request)

        if len(bucket_list) > 0:
            self.t.put(bucket_list)
            self.Preloader.e.set()
            self.made_request = True

    def compute_need(self, i):
        """ Given a current class label, calculate based on batch size how many times
        which classes need to be served in the next batch """
        next_batch_indeces = range(i, i + self.batchsize+1)
        samples_needed = {}
        for i in next_batch_indeces:
            k = (min(i, (i // self.examples_per_batch))) % self.classes
            if k in samples_needed.keys():
                samples_needed[k] += 1
            else:
                samples_needed[k] = 1
        return samples_needed

    def check_stack(self, k, required_samples):
        """ Return true if the audio on stack for class k does only suffice for
        a more serves, false if otherwise """
        remaining_samples = len(self.stack[k])
        if remaining_samples < required_samples:
            return required_samples - remaining_samples
        else:
            return 0

    def make_request(self, k, required_samples):
        """ For class k sample a random corresponding sound file and return a
        tuple of path, label, timestamps required by the preloader. """
        request = []
        for i in range(required_samples):
            sample = self.df[self.df.label == k].sample(n=1)
            path = os.path.join(self.input_dir, sample.path.values[0])
            label = sample.label.values[0]
            request.append((path, label))
        return request

    def inventory(self, i):
        """ Return a dictionary containing the length of audio currently in
        stack for each class. """
        r = {'R':i}
        inv = {k: len(v) for k,v in self.stack.items()}
        return {**r, **inv}
