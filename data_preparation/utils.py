import sqlite3
import urllib.request
import pandas as pd


files_dir = 'storage/top10_german_birds'

conn = sqlite3.connect('storage/db.sqlite')
c = conn.cursor()

def get_recordings_info():
    query = '''SELECT t.id AS taxonomy_id, t.bird_order, t.genus, t.family, t.species, t.german, 
            r.id, r.xeno_canto_id, r.lat, r.long, r.country, r.file, r.time, r.date, 
            r.duration, r.sum_signal, r.timestamps, r.downloaded, r.q
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
    '''
    return pd.read_sql(query, con=conn)

if __name__ == '__main__':
    print(get_recordings_info())