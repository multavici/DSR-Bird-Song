
# Test Run
import pandas as pd
from Spectrogram.spectrograms import mel_s, stft_s
from torch.utils.data import DataLoader
from Datasets.static_dataset import SoundDataset
from get_chunks import get_records_from_classes
from models.bulbul import BulBul



##########################################################################
class_ids =  [5096, 4996, 4993, 4990, 4980]
df = get_records_from_classes(class_ids, 100)

# Check sample distribution:
df.groupby('label').agg({'total_signal':'sum'})

# Split into train and test

##########################################################################




##########################################################################
BATCHSIZE = 64

# Parameters for sample loading
params = {'batchsize' : BATCHSIZE, 
          'window' : 1500, 
          'stride' : 500, 
          'spectrogram_func' : mel_s, 
          'augmentation_func' : None}


ds_test = SoundDataset(df_test, **params)
dl_test = DataLoader(ds_test, BATCHSIZE)

ds_train = SoundDataset(df_train, **params)
dl_train = DataLoader(ds_train, BATCHSIZE)





##########################################################################
time_axis = ds_test[0][0].shape[1]
freq_axis = ds_test[0][0].shape[0]

net = BulBul(time_axis, freq_axis, len(class_ids))




for batch in dl:
    

