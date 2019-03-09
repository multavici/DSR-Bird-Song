'''
This script compares the bird species found in Germany according to Wikipedia
(german_birds.csv) to the species in the Xeno-Canto database and saves the 
german species in the Xeno-Canto database in a csv file.
'''

import pandas as pd
import json

# Get all the bird species in Germany from csv file
birds = pd.read_csv('german_birds.csv')

# Get all bird species in Xeno-Canto db from json text file
with open('taxonomy.txt') as f:
    d_string = f.read()

d = json.loads(d_string)


def in_db(row):
    """Check if species is in dictionary"""
    try:
        d[row['order']][row['family']][row['genus']
                                       ][row['genus']+' '+row['species']]
        return True
    except:
        return False


birds['found'] = birds.apply(lambda row: in_db(row), axis=1)

birds_in_db = birds.loc[birds['found'], [
    'order', 'family', 'genus', 'species']].reset_index(drop=True)
birds_in_db.to_csv('german_birds_in_db.csv')
