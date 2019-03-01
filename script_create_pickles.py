import pandas as pd
import os
import pickle

from Datasets.Preprocessing.utils import load_audio, get_signal, slice_audio
from Spectrogram.spectrograms import stft_s, mel_s
from data_preparation.utils import get_downloaded_records_from_classes

# Check if script is run locally or on server
if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/slices/'
else:
    # script runs locally
    STORAGE_DIR = 'storage/slices/'

# These are the 10 most common birds in the database that are found in Germany
class_ids = [4397, 7091, 6088, 4876, 6265, 4873, 5477, 7232, 6106, 7310]
#seconds_per_class = 100

params = {
          'window' : 5000, 
          'stride' : 1000, 
          }

# Get metadata of samples
df = get_downloaded_records_from_classes(
    class_ids=class_ids, 
    #seconds_per_class=seconds_per_class, 
    min_signal_per_file=params['window'])
print('df created')

# Define the slice function that makes the slices
def prepare_slices(path, timestamps, window, stride):
    audio, sr = load_audio(path)
    signal = get_signal(audio, sr, timestamps)
    audio_slices = slice_audio(signal, sr, window, stride)
    return [mel_s(s) for s in audio_slices]

# Apply the slice function to the samples and save them in the storage folder
for _, row in df.iterrows():
    print('check ', row['id'], 'with label ', row['label'])
    sample_dir = STORAGE_DIR + str(row['id']) + '/'
    if not os.path.exists(sample_dir):
        slices = prepare_slices(row['path'], row['timestamps'], params['window'], params['stride'])
        print('slices made')
        os.makedirs(sample_dir)
        for index, audio_slice in enumerate(slices):
            with open(sample_dir + str(index) + '.pkl', 'wb') as output:
                pickle.dump(audio_slice, output)
            print(f"slice {index} pickled")
        print(f"pickled all slices of {row['id']}")
    else:
        print('already pickled previously')

