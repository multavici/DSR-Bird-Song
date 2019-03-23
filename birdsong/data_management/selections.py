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
    
    def _expected_slices_per_species(self):
        remaining_audio_per_species = sql_selectors.lookup_duration_per_not_downloaded_german_species(self.conn)
        
        def _expected_slices_per_second(duration):
            #TODO: implement better estimator 
            median_signal_per_second = 0.64 # Experimentally established amount
            expectec_signal = duration * median_signal_per_second
            window = 5
            stride = 2.5
            return int(((expectec_signal - window)//stride) + 1)
            
        remaining_audio_per_species['expected_slices'] = remaining_audio_per_species.total_audio.apply(_expected_slices_per_second)
        expected = remaining_audio_per_species[['label', 'expected_slices']].set_index(remaining_audio_per_species.label)
        expected = expected.drop(columns='label')
        return expected
        
    def _active_selection(self, already_available):
        expected = self._expected_slices_per_species()
        assessment = already_available.join(expected, how= 'outer').fillna(0)
        assessment['total_slices'] = assessment.expected_slices + assessment.downloaded_slices
        assessment.sort_values(by='total_slices', inplace=True, ascending=False)
        
        active_selection = assessment.head(self.nr_of_classes)
        return active_selection
    
    def assess_missing(self, already_available):
        active_selection = self._active_selection(already_available)
        if active_selection.total_slices.min() < self.slices_per_class:
            self.slices_per_class = active_selection.total_slices.min()
            print(f'For {self.nr_of_classes} classes only {self.slices_per_class} slices are available.')
        
        active_selection['missing_slices'] = self.slices_per_class - active_selection.downloaded_slices
        return active_selection[['missing_slices']]
        
import sqlite3
from birdsong.data_management.utils import sql_selectors
from birdsong.data_management.management import DatabaseManager


dbm = DatabaseManager('storage')
already_available = dbm.slices_per_species_downloaded()

sel = Selection(dbm.conn, 100, 1000)
expected = sel._expected_slices_per_species()

assessment = sel.assess_material(already_available)

assessment
