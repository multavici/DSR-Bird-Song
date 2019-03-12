"""
This script runs the signal_timestamps function, that gets the duration of each 
recording + all the timestamps of the recoring, and saves the info in the 
database.
"""

import os
import sqlite3

from birdsong.data_preparation.audio_conversion.signal_extraction import signal_timestamps

if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/step1_wav/'
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    STORAGE_DIR = 'storage/step1_wav/'
    DATABASE_DIR = 'storage/db.sqlite'

# Get a list of files that are downloaded
downloaded_files = os.listdir(STORAGE_DIR)
print('list with downloaded files made')
print(len(downloaded_files))

# Get the recording ID's from the filenames
downloaded_ids = [int(x[:-4]) for x in downloaded_files]

# Get all the recordings that were already processed before
conn = sqlite3.connect(DATABASE_DIR)
print('database loaded')
c = conn.cursor()
q = '''
SELECT id FROM recordings
WHERE step1 = 1 AND duration IS NOT NULL
'''
c.execute(q)
processed_ids = [i[0] for i in c.fetchall()]
print('list of already processed recordings')
print(len(processed_ids))

# Remove the already processed recordings from the ones we want to process
to_process = [x for x in downloaded_ids if x not in processed_ids]
print('list of files to process')
print(len(to_process))

# Processing
q = '''
UPDATE recordings
SET duration = ?, sum_signal = ?, timestamps = ? 
WHERE id = ?
'''
batch = []
for i, rec_id in enumerate(to_process):
    rec = str(rec_id) + '.wav'
    print(rec)
    try:
        duration, sum_signal, timestamps = signal_timestamps(
            STORAGE_DIR + rec)
        batch.append((duration, sum_signal, timestamps, rec_id))
        if len(batch) % 50 == 0:
            print(f"batch {i} full")
            c.executemany(q, batch)
            conn.commit()
            batch = []
    except:
        print(f'could not get info of recording {rec}')
        pass
c.executemany(q, batch)
conn.commit()
conn.close()
