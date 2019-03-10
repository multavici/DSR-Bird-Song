'''
This script creates the recordings table in the sqlite database.
'''

import sqlite3
import os

if 'HOSTNAME' in os.environ:
    # script runs on server
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    DATABASE_DIR = 'storage/db.sqlite'

# Create connection to database
conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE recordings
    (id INTEGER PRIMARY KEY,
    xeno_canto_id INTEGER,
    mac_aulay_id INTEGER,
    db TEXT,
    taxonomy_id INTEGER, 
    lat TEXT,
    long TEXT,
    country TEXT,
    file TEXT,
    time TEXT,
    date TEXT,
    quality TEXT)''')
conn.commit()
conn.close()
