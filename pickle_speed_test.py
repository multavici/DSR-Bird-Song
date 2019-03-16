from birdsong.datasets.sequential import SpectralDataset, RandomSpectralDataset
from birdsong.datasets.parallel import RandomSpectralDataset as PRandomSpectralDataset
import pandas as pd
from torch.utils.data import DataLoader
from time import perf_counter as pf
import cProfile

def profileit(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            # Note use of name from outer scope
            prof.dump_stats(name)
            return retval
        return wrapper
    return inner

input_dir_pkl = 'storage/step1_slices'
input_dir_h5 = 'storage/step2_slices'

df_pkl = pd.read_csv('mel_slices_train.csv')
df_h5 = pd.read_csv('mel_slices_train_h5.csv')

ds_seq_classic = SpectralDataset(df_pkl, input_dir_pkl)
ds_seq_random = RandomSpectralDataset(df_pkl, input_dir_pkl)

ds_seq_classic_h5 = SpectralDataset(df_h5, input_dir_h5)
ds_seq_random_h5 = RandomSpectralDataset(df_h5, input_dir_h5)

params = {'batchsize':128,
          'slices_per_class':300, 
          'examples_per_batch':3, 
          'augmentation_func':None, 
          'enhancement_func':None}

ds_par_random_pkl = PRandomSpectralDataset(df_pkl, input_dir=input_dir_pkl, **params)
ds_par_random_h5 = PRandomSpectralDataset(df_h5, input_dir=input_dir_h5, **params)


datasets = {'seq_classic_pkl': ds_seq_classic, 
            'seq_random_pkl': ds_seq_random, 
            'seq_classic_h5': ds_seq_classic_h5, 
            'seq_random_h5': ds_seq_random_h5, 
            'par_random_pkl': ds_par_random_pkl, 
            'par_random_h5': ds_par_random_h5
            }

def test(dl, name):
    @profileit(f'{name}.prof')
    def speed_test(dl):
        start = pf()
        for i, batch in enumerate(dl):
            _ = batch[0]
            print(f'{name}: {pf() - start}')
            start = pf()

            if i == 3:
                break

    speed_test(dl)

def main():
    for name, dataset in datasets.items():
        dl = DataLoader(dataset, 128)
        test(dl, name)
    
    
if __name__ == "__main__":
    main()
    
    
