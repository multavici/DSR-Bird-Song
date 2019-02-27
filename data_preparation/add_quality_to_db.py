import requests
import json
import sqlite3

# Create connection to database
conn = sqlite3.connect("./storage/db.sqlite")
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

# get all families in db
c.execute('''SELECT DISTINCT family FROM taxonomy''')
families = []

for resp in c.fetchall():
    families.append(resp[0])

for family in families:
    data = get_recordings_json(family)
    print(family, len(data['recordings']))

    for recording in data['recordings']:
        c.execute('UPDATE recordings SET q = ? WHERE id = ?',
            (recording['q'], recording['id']))
    conn.commit()