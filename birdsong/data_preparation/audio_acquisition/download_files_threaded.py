'''
This script downloads all recordings of german species from the Xeno-Canto
website.
'''

import sqlite3
import urllib.request
import os
from multiprocessing.pool import ThreadPool

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

query = '''SELECT r.id, r.file
    FROM taxonomy AS t
    JOIN recordings AS r ON t.id = r.taxonomy_id
    WHERE t.german = 1.0 AND downloaded IS NULL'''

recordings = c.execute(query).fetchall()

def download(input_tuple):
    rec_id, path = input_tuple
    try:
        resp = urllib.request.urlretrieve("http:" + path,
                                          STORAGE_DIR + str(rec_id) + ".mp3")
        assert resp[1]['Content-Type'] == 'audio/mpeg', f'file {rec_id} not available'
        print(f'file {rec_id} downloaded')
    except urllib.error.HTTPError:
        print(f'file {rec_id} not found, HTTPError')
        pass


pool = ThreadPool(24)
pool.map(download, recordings)
