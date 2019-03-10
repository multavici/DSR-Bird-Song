'''
This script fills the quality column in the database with the value from the xeno-canto api
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

QUERY_URL = 'https://www.xeno-canto.org/api/2/recordings?query='

# create connection to database
conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()


def get_recordings_json(family):
    url = QUERY_URL + family
    page = requests.get(url)
    data = page.json()
    if data['numPages'] > 1:
        for i in range(2, data['numPages'] + 1):
            page = requests.get(url + '&page=' + str(i))
            data['recordings'] += page.json()['recordings']
    return data


# get all families in db
c.execute('''SELECT DISTINCT family FROM taxonomy''')
families = []

for resp in c.fetchall():
    families.append(resp[0])

# update quality column
for family in families:
    data = get_recordings_json(family)
    print(family, len(data['recordings']))

    for recording in data['recordings']:
        c.execute('UPDATE recordings SET quality = ? WHERE id = ?',
                  (recording['q'], recording['id']))
    conn.commit()

conn.close()
