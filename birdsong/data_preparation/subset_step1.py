'''
This script defines a subset of the data to use for training.
We choose to focus on the most commong bird species found in Germany.
'''

import sqlite3
import os
import pandas as pd
import matplotlib

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

# we choose to only consider recordings that are small enough to handle i.e.
# smaller than 100 seconds
q = '''
SELECT t.id AS species_id, r.xeno_canto_id, t.genus, t.species, r.id, 
    r.scraped_duration AS duration
FROM recordings r 
JOIN taxonomy t ON r.taxonomy_id = t.id
WHERE t.german = 1.0 AND r.scraped_duration < 150 AND r.scraped_duration > 10
'''

df = pd.read_sql(q, conn)

# get distribution of sound per species
species = df.groupby(df.species_id).sum()
species['duration'].sort_values().reset_index(drop=True).plot()

# get ids of the species with more than 1000 seconds of samples
common_species = species[species.duration > 1000].index.tolist()

# get all recordings of these species
df = df.query('species_id in @common_species')

# take only a maximum of 1500 seconds of recording per species, preferably smaller recordings
df = df.sort_values(['species_id', 'duration']).reset_index(drop=True)

for species_id in df.species_id.unique():
    df.loc[df.species_id == species_id,
           'duration_cumsum'] = df[df.species_id == species_id].duration.cumsum()

df_filtered = df.loc[df.duration_cumsum < 1500, :]

ids = tuple(df_filtered['id'].tolist())

q = '''
UPDATE recordings
SET step1 = 1
WHERE id IN 
'''

c.execute(q + str(ids))
conn.commit()
