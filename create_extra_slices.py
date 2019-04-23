import os
import sqlite3
import urllib.request
import os
from multiprocessing.pool import ThreadPool
import librosa
import matplotlib.pyplot as plt
import pickle
import pandas as pd

RECORDINGS_DIR = 'storage/recordings/'
PICKLES_DIR = 'storage/pickles/'
DATABASE_DIR = 'storage/db.sqlite'

top_100_csv = pd.read_csv('app/model/top100_img_codes.csv', names=['id1', 'id2', 'species'])
top_100_genus_species = top_100_csv.species.tolist()
top_100_species = [x.split('_')[1] for x in top_100_genus_species]

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

query = '''SELECT r.id, r.file
    FROM taxonomy AS t
    JOIN recordings AS r ON t.id = r.taxonomy_id
    WHERE t.german = 1.0 AND r.scraped_duration > 100 AND t.species in '''

all_recordings = c.execute(query + str(tuple(top_100_species))).fetchall()
print('all recordings: ', len(all_recordings))
already_sliced = [int(x.split('_')[0]) for x in os.listdir(PICKLES_DIR)]
print('already sliced: ', len(already_sliced))
recordings = [x for x in all_recordings if x[0] not in already_sliced]
print('recordings: ', len(recordings))

import sys
sys.exit()

def download_and_slice(input_tuple):
    rec_id, download_path = input_tuple
    try:
        file_path = RECORDINGS_DIR + str(rec_id) + ".mp3"
        resp = urllib.request.urlretrieve("http:" + download_path, file_path)
        assert resp[1]['Content-Type'] == 'audio/mpeg', f'file {rec_id} not available'
        print(f'file {rec_id} downloaded')
        audio, sr = librosa.load(file_path)

        audio_abs = [abs(x) for x in audio]
        signal_per_second = [sum(audio_abs[i: i + 22050]) for i in range(0, audio.shape[0], 22050)]

        maxdensity, i_start = 0, 0
        for i in range(len(signal_per_second) - 5):
            density = sum(signal_per_second[i:i + 5])
            if density > maxdensity:
                maxdensity = density
                i_start = i
        print(i_start)

        slice_maxwindow = signal_per_second[i_start: i_start + 5]

        slice_signal = audio[i_start * 22050 : (i_start + 5) * 22050]
        
        spect_slice = librosa.feature.melspectrogram(
            slice_signal, sr=22050, n_fft=2048, hop_length=512, n_mels=256, fmin=0, fmax=12000)
        pickle_path = os.path.join(PICKLES_DIR, str(rec_id) + '_0.pkl')
        with open(pickle_path, 'wb+') as f:
            pickle.dump(spect_slice, f)
        os.remove(file_path)
    except urllib.error.HTTPError:
        print(f'file {rec_id} not found, HTTPError')
        pass

pool = ThreadPool(4)
pool.map(download_and_slice, recordings)