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
    
    
        
