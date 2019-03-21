from .utils import sql_selectors


class Selection:
    """ A selection of audio material for a combination of nr of species and
    seconds of audio. 
    
    A selection can also take into account which recordings have already been
    turned into spectrogram slices at previous times. It can then estimate the 
    unknown map D -> S, that is how many slices of signal will be available for 
    a given duration based on previous downloads and thus prioritize
    which files to download to make the selection available locally. """

    def __init__(self, db_conn, nr_of_classes=100, slices_per_class=2000):
        self.conn = db_conn
        self.nr_of_classes = nr_of_classes
        self.slices_per_class = slices_per_class
        
    def _expected_signal_per_second(self, df):
        available_info = df[(~df.sum_signal.isnull()) & (~df.scraped_duration.isnull())]
        if len(available_info) < 100:
            median_signal_per_second = 0.64 # Experimentally established amount
        else:
            median_signal_per_second = (available_info.sum_signal/available_info.scraped_duration).median()
        return median_signal_per_second
        
    def _slices_per_second(self, signal):
        #TODO: Set these with environment variables
        window = 5
        stride = 2.5
        return int(((signal - window)//stride) + 1)
    
    def _recordings_in_selection_still_available(self):
        """ For the specified selection query all available recordings in the 
        database and compute the respectively maximum possible combination of 
        classes and seconds per class and return the recordings that are member
        of this selection"""
        df = sql_selectors.lookup_not_downloaded_german_recordings(self.conn)
        
        exp_signal_per_second = self._expected_signal_per_second(df)
        
        df['expected_slices'] = (df.scraped_duration * exp_signal_per_second).apply(self._slices_per_second)
        distribution = df.groupby('label').expected_slices.sum().sort_values()
        
        selected_classes = distribution.tail(self.nr_of_classes)
        return selected_classes
    
    def missing_recordings(self, already_available):
        return self._all_recordings_in_selection() - already_available
        
        
    def __call__(self):
        return self._all_recordings_in_selection()



import sqlite3
from birdsong.data_management.utils import sql_selectors

conn = sqlite3.connect('storage/db.sqlite')
sel = Selection(conn, nr_of_classes=100)




from birdsong.data_management.management import DatabaseManager

dbm = DatabaseManager('storage')

selected_classes = sel()
already_available = dbm.slices_per_species()

selected_classes.head()

already_available.head()

selected_classes - already_available.loc[selected_classes.index]
