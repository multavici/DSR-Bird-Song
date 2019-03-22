from .utils import sql_selectors
import pandas as pd

class Selection:
    """ A selection of audio material for a combination of nr. of species and
    nr. of signal slices. 
    
    A selection takes into account which recordings have already been
    turned into spectrogram slices at previous times. It can then estimate the 
    unknown map D -> S, that is how many slices of signal will be available for 
    a given duration based on previous downloads and thus prioritize
    which files to download to make the selection available locally. """

    def __init__(self, db_conn, nr_of_classes=100, slices_per_class=1200):
        self.conn = db_conn
        self.nr_of_classes = nr_of_classes
        self.slices_per_class = slices_per_class
        
    def _expected_signal_per_second(self):
        # TODO: implement better estimator 
        median_signal_per_second = 0.64 # Experimentally established amount
        return median_signal_per_second
        
    def _slices_per_second(self, signal):
        #TODO: Set these with environment variables
        window = 5
        stride = 2.5
        return int(((signal - window)//stride) + 1)
    
    def missing_slices(self, still_available, already_available):
        total = pd.concat([still_available, already_available], join='outer', sort=False, axis= 1).fillna(0)
        total['total_available'] = (total.expected_slices + total.available_slices).astype(int)
        total.sort_values(by='total_available', inplace=True)

        selected_classes = total.tail(self.nr_of_classes)
        
        if selected_classes.total_available.min() < self.slices_per_class:
            print(f'For {self.nr_of_classes} classes only {selected_classes.total_available.min()} slices are available.')
            self.slices_per_class = selected_classes.total_available.min()
            
        selected_classes['missing_slices'] = self.slices_per_class - selected_classes.available_slices
        return selected_classes.missing_slices
            
    
    def missing_recordings(self, already_available):
        """ For the specified selection query all not yet downloaded recordings 
        in the database and compute the expected nr of signal slices for each
        of them. """
        df = sql_selectors.lookup_not_downloaded_german_recordings(self.conn)
        
        exp_signal_per_second = self._expected_signal_per_second()
        df['expected_slices'] = (df.scraped_duration * exp_signal_per_second).apply(self._slices_per_second)
        still_available = df.groupby('label').expected_slices.sum().sort_values()
        
        selected_classes_missing = self.missing_slices(still_available, already_available)
        selected_recordings = df[df.label.isin(selected_classes_missing.index.values)]
        
        return selected_recordings
    

        
    def __call__(self):
        pass



import sqlite3
from birdsong.data_management.utils import sql_selectors

conn = sqlite3.connect('storage/db.sqlite')
sel = Selection(conn, nr_of_classes=100)


from birdsong.data_management.management import DatabaseManager
dbm = DatabaseManager('storage', sel)
already_available = dbm.slices_per_species()

selected_recordings = sel.missing_recordings(already_available)

selected_recordings = selected_recordings.sort_values(['label','expected_slices'])

selected_recordings = selected_recordings[selected_recordings.expected_slices > 0]

selected_recordings['cumulative'] = selected_recordings.groupby('label')['expected_slices'].cumsum()


selected_recordings[selected_recordings.cumulative < 1200]
