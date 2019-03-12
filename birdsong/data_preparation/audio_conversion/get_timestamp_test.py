"""
This script runs the signal_timestamps function, that gets the duration of each 
recording + all the timestamps of the recoring, and saves the info in the 
database.

This is a test script for 100 timestamps to see how long it takes.
"""

import os
from .signal_extraction import signal_timestamps
import sqlite3
import time

if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/step1_wav/'
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    STORAGE_DIR = 'storage/step1_wav/'
    DATABASE_DIR = 'storage/db.sqlite'

# Get a list of files that are downloaded
wav_files = os.listdir(STORAGE_DIR)
print('list with downloaded files made')
print(len(wav_files))

start_time = time.time()
# Processing
for wav_file in wav_files[0:100]:
    print(wav_file)
    try:
        duration, sum_signal, timestamps = signal_timestamps(
            STORAGE_DIR + wav_file)
        print(f'duration: {duration}')
        print(f'sum signal: {sum_signal}')
        print(f'timestamps: {timestamps}')
    except:
        print('could not get information')
        pass
print(f'total time: {time.time() - start_time}')
