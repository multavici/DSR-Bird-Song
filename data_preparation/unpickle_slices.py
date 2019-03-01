import pickle
import sqlite3
import os
import pandas as pd

sliced_recordings = os.listdir('storage/slices')
list_recs = []
for dirpath, dirname, filenames in os.walk('storage/slices'):
    for name in filenames:
        path = os.path.join(dirpath, name)
        print(dirpath.split("/")[2])
        


'''
df = pd.DataFrame(list_recs, columns=['id', 'path'])
print(df)

ids = df['id'].unique()
print(ids)
#df = pd.read_sql('select id from recordings where')
'''