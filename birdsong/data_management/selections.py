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
        self.classes_in_selection = self._select_top_k_classes()
        
    def _select_top_k_classes(self):
        """ Selections are always made based on the most popular birds in terms 
        of the total seconds of audio available """
        return sql_selectors.top_k_duration_all_recordings(self.conn, self.nr_of_classes)
    
    def _expected_slices(self):
        """ For the given species selection, check how many seconds of audio are
        still available and compute the expected slices """
        remaining_audio_per_species = sql_selectors.duration_per_not_downloaded_german_species(self.conn, self.classes_in_selection)
        
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
        
    def _inventorize_selection(self, already_available):
        expected = self._expected_slices()
        inventory = expected.join(already_available, how= 'left').fillna(0)
        inventory['total_slices'] = inventory.expected_slices + inventory.downloaded_slices
        inventory.sort_values(by='total_slices', inplace=True, ascending=False)
        return inventory
    
    def assess_missing(self, already_available):
        inventory = self._inventorize_selection(already_available)
        if inventory.total_slices.min() < self.slices_per_class:
            self.slices_per_class = inventory.total_slices.min()
            print(f'For {self.nr_of_classes} classes only {self.slices_per_class} slices are available.')
        
        inventory['missing_slices'] = self.slices_per_class - inventory.downloaded_slices
        return inventory[['missing_slices']]
