import sqlite3
import urllib.request

files_dir = 'storage/top10_german_birds'

conn = sqlite3.connect('storage/db.sqlite')
c = conn.cursor()

top10_german_species = [4397, 7091, 6088, 4876, 6265, 4873, 5477, 7232, 6106, 7310]

query = '''SELECT r.id, r.file
    FROM taxonomy AS t
    JOIN recordings AS r ON t.id = r.taxonomy_id
    WHERE t.id = ?
    LIMIT 100'''

recordings = {}

for species_id in top10_german_species:
    c.execute(query, (species_id,))
    recordings[species_id] = c.fetchall()

for species_id in recordings:
    for rec_id, file_path in recordings[species_id]:
        print(f"start downloading {file_path}")
        try:
            resp = urllib.request.urlretrieve("http:" + file_path,
                'storage/top10_german_birds/' + str(rec_id) + ".mp3")
            assert resp[1]['Content-Type'] == 'audio/mpeg', f'file {rec_id} not available'
            print(f'file {rec_id} downloaded')
        except urllib.error.HTTPError:
            print(f'file {rec_id} not found, HTTPError')
            pass
