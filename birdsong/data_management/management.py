import os
from urllib.request import urlcleanup
from .utils.balanced_split import make_split
from .precomputing_slices import Slicer
from .utils import sql_selectors
from collections import Counter
import pandas as pd
import sqlite3
import datetime
import matplotlib.pyplot as plt

class DatabaseManager(object):
    """ This class bundles the various functions for acquiring, inventorizing, 
    manipulating and serving data and exposes them as easily usable methods."""
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        self.signal_dir = os.path.join(storage_dir, 'signal_slices')
        self.noise_dir = os.path.join(storage_dir, 'noise_slices')
        
        if not os.path.isdir(storage_dir):
            print('Creating empty directory.')
            os.mkdir(storage_dir)
        if not os.path.isdir(self.signal_dir):
            os.mkdir(self.signal_dir)
        if not os.path.isdir(self.noise_dir):
            os.mkdir(self.noise_dir)
        if not os.path.isfile(os.path.join(storage_dir, 'db.sqlite')):
            print('No SQL database built yet, initiating one.')
            #TODO: Actually do that and include all the steps from 
            # database creation so that this entire package could be ported 
            # somewhere else and rebuilt with one command
        
        self.conn = sqlite3.connect(os.path.join(storage_dir, 'db.sqlite'))    
        
        self.SignalSlicer = Slicer(self.signal_dir, type='signal')
        self.NoiseSlicer = Slicer(self.noise_dir, type='noise')
    
    def get_df(self):
        """ Retrieves class name for each slice currently in signal_dir """
        c = self.conn.cursor()
        list_recs = []
        for file in os.listdir(self.signal_dir):
            rec_id = file.split('_')[0]
            species = sql_selectors.lookup_species_by_rec_id(c, rec_id)
            list_recs.append((file, species))   
        df = pd.DataFrame(list_recs, columns=['path', 'label'])
        return df
        
    def slices_per_species(self):
        """ Retrieves Dataframe with class names for currently available slices
        and groups by class """
        df = self.get_df()
        return df.groupby('label').path.count().sort_values()
    
    def download_below_median(self, max_classes=None, max_recordings=10):
        """ Collects class names for which the number of slices is below median
        number of slices and retrieves rec_ids and urls for the first 10 recordings
        for each class that have not been downloaded yet. """
        c = self.conn.cursor()
        balances = self.slices_per_species()
        below_median = balances.index.values[balances < balances.median()]
        if max_classes is None:
            max_classes = len(below_median)
            
        print(f'Fetching recordings for {len(below_median)} classes')
        recordings = []
        running_low = []
        for label in below_median[:max_classes]:
            recordings_for_class = sql_selectors.lookup_recordings_to_download(c, label, max_recordings)
            if len(recordings_for_class) < 10:
                print(f'Running low on {label}')
                running_low.append(label)
            recordings += recordings_for_class
        print(f'Selected {len(recordings)} recordings for slicing')
        rec_ids_to_download = list(map((lambda x: str(x[0])), recordings))
        
        # Handle recordings in bunches of 24 to avoid filling tmp too much:
        at_a_time = 24
        for bunch in [recordings[i:i+at_a_time] for i in range(0, len(recordings), at_a_time)]:
            self.SignalSlicer(bunch)
            urlcleanup()
        
        # Update DB:
        sql_selectors.set_downloaded
        sql_selectors.set_slices(c, rec_ids_to_download)
        
        # Log update of slices:
        new_balances = self.slices_per_species()
        differences = new_balances - balances
        self._plot_difference(balances, differences)

    def _plot_slices_before_after_downloading(self, balances, differences):
        """ Stores a plot showing the class distribution before and after 
        downloading new slices """
        now = datetime.datetime.now()
        df = pd.DataFrame({'before' : balances, 'added' : differences})
        df.plot(kind='bar', stacked=True)
        import matplotlib.pyplot as plt
        plt.savefig(f'Class balances {now}.pdf', bbox_inches = "tight")
    
    def seconds_per_species_local_remote(self):
        """ This compares the total seconds of audio material available for each
        species that has already been downloaded vs what is still available.   
        """
        downloaded = sql_selectors.lookup_downloaded_german_recordings(self.conn)
        downloaded['downloaded'] = 1
        
        not_downloaded = sql_selectors.lookup_not_downloaded_german_recordings(self.conn)
        not_downloaded['downloaded'] = 0
        df = pd.concat([downloaded, not_downloaded])
        return df
    
    def _plot_seconds_per_species_local_remote(self, df):
        now = datetime.datetime.now()
        dl = df[df.downloaded == 1].groupby('label').scraped_duration.sum().sort_values()
        ndl = df[df.downloaded == 0].groupby('label').scraped_duration.sum().sort_values()
        plt.figure(figsize=(50,150))
        p2 = plt.barh(ndl.index.values, ndl)
        p1 = plt.barh(dl.index.values, dl)
        plt.legend((p1[0], p2[0]), ('Downloaded', 'Not Downloaded'))
        plt.savefig(f'Downloaded vs not Downloaded {now}', bbox_inches = "tight")

    def resample_df(self, df, samples_per_class):
        """ Up- or downsample a dataframe by randomly picking a fixed number of 
        samples for each class """
        out = df.groupby('label').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
        return out.sample(frac=1).reset_index(drop=True)
    
    def split(self):
        """
        from birdsong.data_preparation.balanced_split import make_split
        import pandas as pd

        df = pd.read_csv('label_table.csv').rename(columns={'id':'rec_id'})
        df.groupby(['label', 'rec_id']).count()#.sort_values('path')

        train, test = make_split(df, 20)
        test.to_csv('mel_slices_test.csv')
        train.to_csv('mel_slices_train.csv')
        """
        pass
        
    def inventory(self):
        files = os.listdir(self.signal_dir)
        rec_ids = [file.split('_')[0] for file in files]
        self.counts = {k: v for k,v in zip(Counter(rec_ids).keys(), Counter(rec_ids).values())}
        
    def make_some_noise(self):
        c = self.conn.cursor()
        balances = self.slices_per_species()
        labels = balances.index.values
            
        recordings = []
        for label in labels:
            recordings_for_class = sql_selectors.lookup_recordings_for_noise(c, label, 1)
            recordings += recordings_for_class
        print(f'Selected {len(recordings)} recordings for noise slicing')
        rec_ids_to_download = list(map((lambda x: str(x[0])), recordings))
        
        # Handle recordings in bunches of 24 to avoid filling tmp too much:
        at_a_time = 24
        for bunch in [recordings[i:i+at_a_time] for i in range(0, len(recordings), at_a_time)]:
            self.NoiseSlicer(bunch)
            urlcleanup()



"""

dbm = DatabaseManager('storage')

slices_per_species = dbm.slices_per_species()

len(slices_per_species)

df = dbm.seconds_per_species_local_remote()

downloaded = df[df.downloaded == 1].groupby('label').scraped_duration.sum().sort_values()
len(downloaded)

downloaded.loc['tyto_alba']
slices_per_species.loc['tyto_alba']

slices_seconds = pd.concat([slices_per_species, downloaded], axis = 1)
slices_per_second = slices_seconds.path / slices_seconds.scraped_duration

slices_seconds.plot(kind='barh', figsize=(10, 80))
slices_per_second.plot()
"""
