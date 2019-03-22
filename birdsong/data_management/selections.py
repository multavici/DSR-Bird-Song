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

    def __init__(self, db_conn, nr_of_classes, slices_per_class):
        self.conn = db_conn
        self.nr_of_classes = nr_of_classes
        self.slices_per_class = slices_per_class
        
    def _expected_signal_per_second(self):
        #TODO: implement better estimator 
        median_signal_per_second = 0.64 # Experimentally established amount
        return median_signal_per_second
        
    def _slices_per_second(self, signal):
        #TODO: Set these with environment variables
        window = 5
        stride = 2.5
        return int(((signal - window)//stride) + 1)
    
    def _missing_slices(self, still_available, already_available):
        total = pd.concat([still_available, already_available], join='outer', sort=False, axis= 1).fillna(0)
        total['total_available'] = (total.expected_slices + total.available_slices).astype(int)
        total.sort_values(by='total_available', inplace=True)

        selected_classes = total.tail(self.nr_of_classes)
        
        #Store for future use
        self.classes_in_selection = selected_classes.index
        
        if selected_classes.total_available.min() < self.slices_per_class:
            print(f'For {self.nr_of_classes} classes only {selected_classes.total_available.min()} slices are available.')
            self.slices_per_class = selected_classes.total_available.min()
            
        selected_classes['missing_slices'] = self.slices_per_class - selected_classes.available_slices
        selected_classes = selected_classes[selected_classes.missing_slices > 0]
        return selected_classes.missing_slices
    
    def assess_missing_recordings(self, already_available):
        """ For the specified selection query all not yet downloaded recordings 
        in the database and compute the expected nr of signal slices for each
        of them. """
        df = sql_selectors.lookup_not_downloaded_german_recordings(self.conn)
        
        exp_signal_per_second = self._expected_signal_per_second()
        df['expected_slices'] = (df.scraped_duration * exp_signal_per_second).apply(self._slices_per_second)
        still_available = df.groupby('label').expected_slices.sum().sort_values()
        
        selected_classes_missing = self._missing_slices(still_available, already_available)
        nr_needed = selected_classes_missing.sum()
        print(f'Need {int(nr_needed)} more slices for {len(selected_classes_missing)} class(es), that is {nr_needed * 432.2 / 1024} MB.')
        
        selected_recordings = df[df.label.isin(selected_classes_missing.index.values)]
        selected_recordings['missing'] = list(map(lambda x: selected_classes_missing.loc[x], selected_recordings.label))
        selected_recordings = selected_recordings.sort_values(['label','expected_slices'])
        selected_recordings = selected_recordings[selected_recordings.expected_slices > 0]

        selected_recordings['cumulative'] = selected_recordings.groupby('label')['expected_slices'].cumsum()
    
        selected_recordings = selected_recordings[selected_recordings.cumulative <= selected_recordings.missing + 5]
        print(f'Filling up requires downloading {len(selected_recordings)} more recordings.')
        selected_recordings.file = selected_recordings.file.apply(lambda x: 'http:' + x)
        self.missing_recordings = list(selected_recordings[['id', 'file']].itertuples(index=False, name=None))
            
    def __call__(self, already_available):
        pass
        
