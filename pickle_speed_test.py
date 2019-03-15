from birdsong.datasets.sequential import SpectralDataset, RandomSpectralDataset
import pandas as pd
from torch.utils.data import DataLoader
from time import perf_counter as pf()
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

input_dir = 'storage/step1_slices'


df_test = pd.read_csv('mel_slices_test')
df_train = pd.read_csv('mel_slices_train')

ds_test = SpectralDataset(df_test, input_dir)
ds_train = RandomSpectralDataset(df_train, input_dir)

dl_test = DataLoader(ds_test, 128)
dl_train = DataLoader(ds_train, 128)


def test(dl, name):
    @profileit(name)
    def speed_test(dl):
        start = pf()
        for i, batch in enumerate(dl):
            _ = batch[0]
            print(pf() - start)
            if i == 300:
                break
    
    speed_test(dl)

def main():
    test(dl_test, 'test_loader.prof')
    test(dl_train, 'train_loader.prof')

    
if __name__ == "__main__":
    main()
    
    
