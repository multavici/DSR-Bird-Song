"""
This script converts mp3 files from different sample rates and amount of channels
to wav files with a fixed sample rate of 22050 and 1 channel (mono)
"""

import os
import sqlite3
import librosa
from multiprocessing.pool import ThreadPool

if 'HOSTNAME' in os.environ:
    # script runs on server
    INPUT_DIR = '/storage/step1/'
    OUTPUT_DIR = '/storage/step1_wav/'
else:
    # script runs locally
    INPUT_DIR = 'storage/step1/'
    OUTPUT_DIR = 'storage/step1_wav/'

mp3_files = os.listdir(INPUT_DIR)


def convert(mp3_file):
    y, sr = librosa.core.load(INPUT_DIR + mp3_file)
    librosa.output.write_wav(OUTPUT_DIR + mp3_file[:-4] + '.wav', y, sr)
    print(f'file {mp3_file} converted')


pool = ThreadPool(24)
pool.map(convert, mp3_files)
