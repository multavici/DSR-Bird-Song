#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:17:13 2019

@author: tim
"""
import os
import pickle
import librosa
import numpy as np
import urllib
from urllib.request import urlretrieve, urlcleanup
from multiprocessing.pool import ThreadPool
from .utils.signal_extraction import signal_noise_separation
from .utils.io import slice_audio
from .utils.spectrograms import mel_s
import random
import warnings

class Slicer:
    def __init__(self, output_dir, window=5000, stride=2500, type='signal'):
        self.output_dir = output_dir
        self.window = window
        self.stride = stride
        self.type = type
        assert self.type in ['signal', 'noise'], "Type can either be 'signal' or 'noise'"

    def _temp_download(self, url):
        """ Download audio to temporary folder and extract noise and signal """
        temp_file = urlretrieve(url)[0]
        audio, sr = librosa.load(temp_file, sr = 22050, mono=True)
        signal, noise = signal_noise_separation(audio)
        return signal, noise, sr
    
    def _spec_slices(self, audio, sr):
        """ Take audio and return mel spectrogram slices """
        audio_slices = slice_audio(audio, sr, self.window, self.stride)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            spec_slices = [mel_s(s, n_mels=256, fmax=12000) for s in audio_slices]
        return spec_slices
    
    def _save_slices(self, slices, rec_id):
        """ Save slices as pickles """
        for index, audio_slice in enumerate(slices):
            slice_name = str(rec_id) + '_' + str(index) + '.pkl'
            with open(os.path.join(self.output_dir, slice_name), 'wb') as output:
                pickle.dump(audio_slice, output)
    
    def download_and_slice(self, rec_id_url_tuple):
        """ Bundles the previous private methods to preprocess one tuple of 
        rec_id and url, download to /tmp and store slices as pickles. """
        rec_id, url = rec_id_url_tuple
        try:
            signal, noise, sr = self._temp_download(url)
        except:# urllib.error.HTTPError: NoBackEndError
            print('Problem with recording',  rec_id)
            return
        if self.type == 'signal':
            if len(signal) >= (self.window / 1000) * sr:
                spec_slices = self._spec_slices(signal, sr)
            else:
                return
        if self.type == 'noise':
            if len(noise) >= (self.window / 1000) * sr:
                spec_slices = self._spec_slices(noise, sr)
                if len(spec_slices) > 4:
                    spec_slices = random.sample(spec_slices, 4)
            else:
                return
        self._save_slices(spec_slices, rec_id)
                    
    def __call__(self, list_of_tuples):
        """ Accepts a list of (rec_id, url) tuples that it will process in a 
        threadpool. """
        pool = ThreadPool(min(12, len(list_of_tuples)))
        pool.map(self.download_and_slice, list_of_tuples)
        urlcleanup()
