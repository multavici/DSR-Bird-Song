"""
This script creates a CSV file with the path to all pickled spectograms of slices
of audio + the corresponding id of the recording and the label.
"""

import os
import sqlite3
import pandas as pd


if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/step1_slices/'
    DATABASE_DIR = '/storage/db.sqlite'
    FILES_DIR = '/storage/label_tables/'
else:
    # script runs locally
    STORAGE_DIR = 'storage/step1_slices/'
    DATABASE_DIR = 'storage/db.sqlite'
    FILES_DIR = 'storage/label_tables/'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

files = os.listdir(STORAGE_DIR)
ids = [x.split("_")[0] for x in files]
paths_ids = [(x, int(x.split("_")[0])) for x in files]

slices = pd.DataFrame(paths_ids, columns=['path', 'rec_id'])

q = '''
SELECT r.id rec_id, t.genus, t.species
FROM recordings r
JOIN taxonomy t
    ON r.taxonomy_id = t.id
WHERE r.id IN 
'''

recordings = pd.read_sql(q + str(tuple(ids)), conn)

label_table = slices.merge(recordings)

label_table['label'] = label_table['genus'] + " " + label_table['species']
label_table['path'] = label_table['path'].apply(lambda x: str(x)) + ".wav"
label_table.drop(columns=['genus', 'species'], inplace=True)

label_table.to_csv('storage/label_tables/label_table.csv', index=False)
