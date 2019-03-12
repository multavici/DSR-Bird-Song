"""
This script takes as input a list of species id's, a window size and a stride
size. It retrieves in the database the paths of all recordings that belong to 
the specified species and the timestamps of the signal parts in the recording. 
The script then chops up the signal part of the recordings to slices with length
window size and stride stride. 

For each recording a new folder is made and within that folder all slices are
saved as pickle files.
"""

import pandas as pd
import os
import pickle
import time
import sqlite3

from birdsong.datasets.tools.io import load_audio, get_signal, slice_audio
from birdsong.datasets.tools.spectrograms import mel_s

# Check if script is run locally or on server
if 'HOSTNAME' in os.environ:
    # script runs on server
    INPUT_DIR = '/storage/step1_wav/'
    OUTPUT_DIR = '/storage/step1_slices/'
    DATABASE_DIR = '/storage/db.sqlite'
    TABLE_DIR = '/storage/label_tables/'
else:
    # script runs locally
    INPUT_DIR = 'storage/step1_wav/'
    OUTPUT_DIR = 'storage/step1_slices/'
    DATABASE_DIR = 'storage/db.sqlite'
    TABLE_DIR = 'storage/label_tables/'

WINDOW = 5000
STRIDE = 2500

conn = sqlite3.connect(DATABASE_DIR)

q = '''
SELECT r.id, t.genus, t.species, r.timestamps
FROM recordings r
JOIN taxonomy t
    ON r.taxonomy_id = t.id
WHERE downloaded = 1.0 AND step1 = 1 AND duration > 5
'''

df = pd.read_sql(q, conn)
conn.close()

df['id'] = df['id'].apply(str)
df['label'] = df['genus'] + " " + df['species']
df['path'] = INPUT_DIR + df['id'] + '.wav'

df.drop(columns=['genus', 'species'], inplace=True)


def prepare_slices(path, timestamps, window, stride):
    audio, sr = load_audio(path)
    signal = get_signal(audio, sr, timestamps)
    audio_slices = slice_audio(signal, sr, window, stride)
    return [mel_s(s, n_mels=256, fmax=12000) for s in audio_slices]


label_table = open(TABLE_DIR + 'label_table.csv', 'w+')

start = time.time()
# Apply the slice function to the samples and save them in the storage folder
for _, row in df.iterrows():
    print('check ', row['id'], 'with label ', row['label'])
    print('timestamps: ' + row['timestamps'])

    slices = prepare_slices(row['path'], row['timestamps'], WINDOW, STRIDE)
    print('slices made')

    for index, audio_slice in enumerate(slices):
        slice_name = row['id'] + '_' + str(index) + '.pkl'
        with open(OUTPUT_DIR + slice_name, 'wb') as output:
            pickle.dump(audio_slice, output)
        label_table.write(slice_name + "," +
                          row['id'] + "," + row['label'] + '\n')
        print(f"slice {index} pickled")

    print(f"pickled all slices of {row['id']}")

print(f'end time: {time.time() - start}')

label_table.close()
