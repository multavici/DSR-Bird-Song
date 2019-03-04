import pandas as pd 
import os
import sqlite3

recordings = os.listdir('storage/top10_german_birds')
l = []

conn = sqlite3.connect('storage/db.sqlite')
c = conn.cursor()

for dirname, dirs, files in os.walk('storage/slices'):
    for file in files:
        rec_id = dirname.split("/")[-1]
        c.execute('select t.genus, t.species from recordings as r join taxonomy as t on r.taxonomy_id = t.id where r.id = ?', (rec_id,))
        fetch = c.fetchone()
        label = fetch[0] + "_" + fetch[1]
        path = os.path.join(dirname, file)
        l.append((rec_id, path, label))


df = pd.DataFrame(l, columns=['rec_id', 'path', 'label'])
print(df)
df.to_csv('slices_and_labels2.csv')
