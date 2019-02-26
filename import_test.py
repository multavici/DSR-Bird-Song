
# Test Run
import pandas as pd
from Spectrogram.spectrograms import mel_s
from torch.utils.data import DataLoader
df = pd.read_csv('Testing/test_df.csv')


params = {'batchsize' : 64, 
          'window' : 1500, 
          'stride' : 500, 
          'spectrogram_func' : mel_s, 
          'augmentation_func' : None}


from Datasets.static_dataset import SoundDataset

ds = SoundDataset(df, **params)

len(ds)

dl = DataLoader(ds, 64)


for batch in dl:
    print(batch[0].shape, batch[1])


# To visualize training samples:
import matplotlib.pyplot as plt
plt.imshow(ds[1][0])