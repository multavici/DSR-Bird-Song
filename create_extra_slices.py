import os
import sqlite3
import urllib.request
import os
from multiprocessing.pool import ThreadPool
import librosa
import matplotlib.pyplot as plt
import pickle

RECORDINGS_DIR = 'storage/recordings/'
PICKLES_DIR = 'storage/pickles/'
DATABASE_DIR = 'storage/db.sqlite'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

query = '''SELECT r.id, r.file
    FROM taxonomy AS t
    JOIN recordings AS r ON t.id = r.taxonomy_id
    WHERE t.german = 1.0'''

recordings = c.execute(query).fetchall()

batch_1 = recordings[:33_000]
batch_2 = recordings[33_000:66_000]
batch_3 = recordings[66_000:]

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
pool.map(download_and_slice, batch_1)