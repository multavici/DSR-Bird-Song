import os
import json
import pandas as pd
import sqlite3

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

files = os.listdir(STORAGE_DIR)

d_list = []
for file_string in files:
    print(file_string)
    with open(STORAGE_DIR + file_string, 'r') as f:
        text = f.read()
    amount = text.count('{')
    s = text
    for i in range(amount):
        start = s.find("{") + 1
        s = s[start:]
        end = s.find("}")
        d_list.append(json.loads('{' + s[:end] + '}'))


df = pd.DataFrame(d_list)
df.drop(df.index[df.channels == '0'], inplace=True)
df.drop(df.index[df['sr'].isnull()], inplace=True)

df['sr'] = df['sr'].apply(lambda x: int(x.split(" ")[0]))
df['duration'] = df['duration'].apply(lambda x: int(x.split(" ")[0]))
df['channels'] = df['channels'].apply(lambda x: int(x.split(" ")[0]))
df['channels'] = df['bitrate'].apply(lambda x: int(x.split(" ")[0]))

taxonomy_table = pd.read_sql('select * from taxonomy', conn)

def get_taxonomy_ids(s):
    amount = s.count("(")
    l = []
    for i in range(amount):
        start = s.find("(") + 1
        s = s[start:]
        end = s.find(")")
        genus, species = s[:end].split(" ")
        tax_id = taxonomy_table.query('genus == @genus.lower() and species == @species').iloc[0].loc['id']
        l.append(tax_id)
    return l

df['background_species'] = df['background_species'].apply(get_taxonomy_ids)