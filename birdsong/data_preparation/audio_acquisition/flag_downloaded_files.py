'''
This script flags the downloaded files in the database
'''

import sqlite3
import os

if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/all_german_birds/'
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    STORAGE_DIR = 'storage/all_german_birds/'
    DATABASE_DIR = 'storage/db.sqlite'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

file_list = os.listdir(STORAGE_DIR)
id_list = [x[:-4] for x in file_list]

c.execute('UPDATE recordings SET downloaded = 1.0 WHERE id IN ' +
          str(tuple(id_list)))
conn.commit()
conn.close()
