"""
This script converts the leftover mp3 files from different sample rates and 
amount of channels to wav files with a fixed sample rate of 22050 and 1 channel
(mono)
"""

import os
import sqlite3
import librosa

if 'HOSTNAME' in os.environ:
    # script runs on server
    INPUT_DIR = '/storage/step1/'
    OUTPUT_DIR = '/storage/step1_wav/'
else:
    # script runs locally
    INPUT_DIR = 'storage/step1/'
    OUTPUT_DIR = 'storage/step1_wav/'

mp3_files = os.listdir(INPUT_DIR)
wav_files = os.listdir(OUTPUT_DIR)

mp3_ids = [x[:-4] for x in mp3_files]
wav_ids = [x[:-4] for x in wav_files]


def convert(mp3_file):
    try:
        y, sr = librosa.core.load(INPUT_DIR + mp3_file)
        librosa.output.write_wav(OUTPUT_DIR + mp3_file[:-4] + '.wav', y, sr)
        print(f'file {mp3_file} converted')
    except:
        print(f'problem with file {mp3_file}: not converted')


to_convert = [x for x in mp3_ids if x not in wav_ids]

to_convert_mp3 = [x + '.mp3' for x in to_convert]

for f in to_convert_mp3:
    convert(f)
