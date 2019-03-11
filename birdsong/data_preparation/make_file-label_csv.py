'''
This script makes a csv file with recording paths, their label, the duration,
the total amount of signal and the the timestamps of the signal.
This info is used in the dataloader of the model.
'''

import os
import sqlite3

if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/step1/'
    DATABASE_DIR = '/storage/db.sqlite'
    FILES_DIR = '/storage/label_tables/'
else:
    # script runs locally
    STORAGE_DIR = 'storage/step1/'
    DATABASE_DIR = 'storage/db.sqlite'
    FILES_DIR = 'storage/label_tables/'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

files = os.listdir(STORAGE_DIR)
ids = [x[:-4] for x in files]

q = '''
SELECT r.id, t.species, r.duration, r.sum_signal, r.timestamps
FROM recordings r
JOIN taxonomy t
    ON r.taxonomy_id = t.id
WHERE r.id IN 
'''

c.execute(q + str(tuple(ids)))
