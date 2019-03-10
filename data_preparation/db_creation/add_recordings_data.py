'''
This script gets the metadata of all the recordings available via the Xeno-Canto 
api and saves it in the recordings table of the sqlite database.

We can get the most recordings per request by querying on the family.
'''

import requests
import json
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


def get_recordings_json(family):
    url = "https://www.xeno-canto.org/api/2/recordings?query="+family
    page = requests.get(url)
    data = page.json()
    if data['numPages'] > 1:
        for i in range(2, data['numPages'] + 1):
            page = requests.get(url + '&page=' + str(i))
            data['recordings'] += page.json()['recordings']
    return data


# get families
c.execute('''SELECT DISTINCT family FROM taxonomy''')
families = []

for resp in c.fetchall():
    families.append(resp[0])

for family in families:
    data = get_recordings_json(family)
    print(family, len(data['recordings']))

    for recording in data['recordings']:
        # look up taxonomy id
        c.execute('''SELECT id FROM taxonomy WHERE genus = ? AND species = ?''',
                  (recording['gen'].lower(), recording['sp'].lower()))
        try:
            taxonomy_id = c.fetchone()[0]
        except:
            print(f"{recording['id']}: species {recording['sp']} not found")
            break
        # add to recordings table
        c.execute('''INSERT INTO recordings (xeno_canto_id, db, taxonomy_id, 
            lat, long, country, file, time, date, quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (recording['id'],
                   'xeno_canto',
                   taxonomy_id,
                   recording['lat'],
                   recording['lng'],
                   recording['cnt'].lower(),
                   recording['file'],
                   recording['time'],
                   recording['date'],
                   recording['q']))
        assert c.rowcount == 1, f'{recording["id"]}insertion failed'
    conn.commit()

conn.close()
