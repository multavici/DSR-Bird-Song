
class LabelEncoder:
    """ A custom label encoder to work on pandas series """
    def __init__(self, label_series):
        self.original_series = label_series

    def encode(self):
        """ Works on a pandas series of labels and returns a label encoded version
        plus a dictionary of codes """
        self.codes = {}
        encoded_col = self.original_series.copy()
        for i, label in enumerate(sorted(self.original_series.drop_duplicates())):
            encoded_col.loc[encoded_col == label] = i
            self.codes[i] = label
        return encoded_col

    def _decode(self, code):
        """ Returns the original label for a single encoded instance """
        return self.codes[code]

    def label_decoder(self, list_of_codes):
        """ Decodes a list of codes based on previously generated code dictionary"""
        try:
            return [_decode(code) for code in list_of_codes]
        except:
            print("I have not yet encoded anything - call my .encode method")
