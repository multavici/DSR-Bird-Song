'''
This script scrapes the detail pages of the recordings on the Xeno-Canto website
for information that is not available through the api.

Threads are used to speed up the process. Each thread saves the info in a .json
file
'''

import os
import sqlite3
import pandas as pd
from multiprocessing.pool import ThreadPool
import threading
import json

if 'HOSTNAME' in os.environ:
    # script runs on server
    STORAGE_DIR = '/storage/extra_info/'
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    STORAGE_DIR = 'storage/extra_info/'
    DATABASE_DIR = 'storage/db.sqlite'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

c.execute('SELECT id, xeno_canto_id FROM recordings_backup')
recordings = c.fetchall()


def download(input_tuple):
    rec_id, xeno_canto_id = input_tuple
    tables = pd.read_html('https://www.xeno-canto.org/' + str(xeno_canto_id))
    circumstances, mp3_info, processed_info = tables[2], tables[3], tables[4]
    d = {
        'id': rec_id,
        'xeno_canto_id': xeno_canto_id,
        'background_species': circumstances.iloc[8, 1],
        'duration': mp3_info.iloc[0, 1],
        'sr': mp3_info.iloc[1, 1],
        'bitrate': mp3_info.iloc[2, 1],
        'channels': mp3_info.iloc[3, 1],
        'type': processed_info.iloc[0, 1],
        'volume': processed_info.iloc[1, 1],
        'speed': processed_info.iloc[2, 1],
        'pitch': processed_info.iloc[3, 1],
        'length': processed_info.iloc[4, 1],
        'no_notes': processed_info.iloc[5, 1],
        'variable': processed_info.iloc[6, 1]
    }
    with open(STORAGE_DIR + str(threading.get_ident()) + '.json', 'a+') as f:
        f.write(json.dumps(d))
        f.write('\n')


pool = ThreadPool(24)
pool.map(download, recordings)
