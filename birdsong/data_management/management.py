import os
from urllib.request import urlcleanup
from .utils.balanced_split import make_split
from .utils import sql_selectors
from .precomputing_slices import Slicer
from .selections import Selection
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
        self.Selection = Selection
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
    
    def make_selection(self, nr_of_classes=100, slices_per_class=1200):
        self.Selection = Selection(self.conn, nr_of_classes, slices_per_class)
        already_available = self.slices_per_species()
        self.Selection.assess_missing_recordings(already_available)
        
    def get_df(self):
        classes_in_selection = self.Selection.classes_in_selection
        all = self.inventory_df()
        available_in_selection = all[all.label.isin(classes_in_selection)].reset_index(drop=True)
        ideal = self.Selection.nr_of_classes * self.Selection.slices_per_class
        slices_available = available_in_selection.groupby('label').count().sum().values
        if slices_available < ideal:
            print(f"We are {ideal - slices_available} slices short of the Selection. \
            You can call the method 'download_missing' to fill them up if more are available.")
            
        return available_in_selection
        
    def inventory_df(self):
        """ Retrieves class name for each slice currently in signal_dir 
        and returns of df with the file name for each recording and its 
        associated label. """
        c = self.conn.cursor()
        list_recs = []
        for file in os.listdir(self.signal_dir):
            if file.endswith('.pkl'):
                rec_id = file.split('_')[0]
                species = sql_selectors.lookup_species_by_rec_id(c, rec_id)
                list_recs.append((file, species))   
        df = pd.DataFrame(list_recs, columns=['path', 'label'])
        return df
        
    def slices_per_species(self):
        """ Retrieves Dataframe with class names for currently available slices
        and groups by class """
        df = self.inventory_df().rename(columns={'path':'available_slices'})
        return df.groupby('label').available_slices.count().astype(int).sort_values()
        
    def download_missing(self):
        balances = self.slices_per_species()
        to_download = self.Selection.missing_recordings
        if len(to_download) == 0:
            print('Nothing to download, Selection compltete.')
            return
        
        self._download_threaded(to_download)
        # Log update of slices:
        new_balances = self.slices_per_species()
        differences = new_balances - balances
        print(differences[differences > 0])
        
        #self._plot_slices_before_after_downloading(balances, differences)
    
    def _download_threaded(self, recordings):
        # Handle recordings in bunches of 24 to avoid filling tmp too much:
        at_a_time = 24
        total = len(recordings)
        print(f'Downloading {total} recording(s)')
        for iteration, bunch in enumerate([recordings[i:i+at_a_time] for i in range(0, len(recordings), at_a_time)]):
            self.SignalSlicer(bunch)
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(100 * iteration // total)
            bar = fill * filledLength + '-' * (100 - filledLength)
            print('\r%s |%s| %s%% %s' % ('', bar, percent, ''), end = '\r')
            if iteration == total: 
                print()
        
        print('Done downloading!')
        # Update DB:
        c = self.conn.cursor()
        rec_ids_to_download = list(map((lambda x: str(x[0])), recordings))
        sql_selectors.set_downloaded(c, rec_ids_to_download)
        return
    
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

    def resample_df(self, df, samples_per_class):
        """ Up- or downsample a dataframe by randomly picking a fixed number of 
        samples for each class """
        out = df.groupby('label').apply(lambda x: x.sample(n=samples_per_class, replace=True)).reset_index(drop=True)
        return out.sample(frac=1).reset_index(drop=True)
    
    def slices_per_downloaded_recording(self):
        files = os.listdir(self.signal_dir)
        rec_ids = [file.split('_')[0] for file in files]
        return {k: v for k,v in zip(Counter(rec_ids).keys(), Counter(rec_ids).values())}
        
    def make_some_noise(self):
        c = self.conn.cursor()
        balances = self.slices_per_species()
        labels = balances.index.values
            
        # Get noise for all species currently downloaded
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
    def _plot_slices_before_after_downloading(self, balances, differences):
        now = datetime.datetime.now()
        df = pd.DataFrame({'before' : balances, 'added' : differences})
        df.plot(kind='barh', stacked=True)
        plt.savefig(f'Class balances {now}.pdf', bbox_inches = "tight")
    
    def _plot_seconds_per_species_local_remote(self, df):
        now = datetime.datetime.now()
        dl = df[df.downloaded == 1].groupby('label').scraped_duration.sum().sort_values()
        ndl = df[df.downloaded == 0].groupby('label').scraped_duration.sum().sort_values()
        plt.figure(figsize=(50,150))
        p2 = plt.barh(ndl.index.values, ndl)
        p1 = plt.barh(dl.index.values, dl)
        plt.legend((p1[0], p2[0]), ('Downloaded', 'Not Downloaded'))
        plt.savefig(f'Downloaded vs not Downloaded {now}', bbox_inches = "tight")
    """
