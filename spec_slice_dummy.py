#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:51:45 2019

@author: tim
"""
import pandas as pd
import os
import pickle

from Datasets.Preprocessing.utils import load_audio, get_signal, slice_audio
from Spectrogram.spectrograms import stft_s, mel_s
from data_preparation.get_chunks import get_records_from_classes
from data_preparation.utils import get_recordings_info



#class_ids = [6088, 3912, 4397] #, 7091] #, 4876, 4873, 5477, 6265, 4837, 4506] # all have at least 29604 s of signal, originally 5096, 4996, 4993, 4990, 4980
class_ids = [4397, 7091, 6088, 4876, 6265, 4873, 5477, 7232, 6106, 7310]
seconds_per_class = 100


params = {
          'window' : 5000, 
          'stride' : 1000, 
          }

#df_all = get_recordings_info()
#df = df_all.query('taxonomy_id in @class_ids and downloaded == 1.0')
#print(df)

# Get metadata of samples
df = get_records_from_classes(
    class_ids=class_ids, 
    seconds_per_class=seconds_per_class, 
    min_signal_per_file=params['window'])
print('df created')

'''
# If working locally, download the specific files
if not 'HOSTNAME' in os.environ:
    from data_preparation.download_recording_by_id import download_recordings
    download_recordings(df['xeno-canto_id'].tolist())
'''

def prepare_slices(path, timestamps, window, stride):
    audio, sr = load_audio(path)
    signal = get_signal(audio, sr, timestamps)
    audio_slices = slice_audio(signal, sr, window, stride)
    return [mel_s(s) for s in audio_slices]

for _, row in df.iterrows():
    slices = prepare_slices(row['path'], row['timestamps'], params['window'], params['stride'])
    rec_dir = 'storage/slices/' + str(row['id']) 
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    for index, audio_slice in enumerate(slices):
        output = open(rec_dir + '/' + str(index) + '.pkl', 'wb')
        pickle.dump((audio_slice, row['label']), output)
        print("slice pickled")
    print(f"pickled all slices of {row['id']}")

