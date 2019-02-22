#%%

import sqlite3
import json
import pandas as pd
import os

# Create connection to database, always run commands from main directory
db_dir = os.path.join(os.getcwd(), 'storage', 'db.sqlite')
print(db_dir)
conn = sqlite3.connect(db_dir)

def species_df():
    query = '''SELECT t.bird_order, t.family, t.genus, t.species, t.german,
            COUNT(r.id)
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        GROUP BY t.bird_order, t.family, t.genus, t.species
        ORDER BY COUNT(r.id) DESC'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def orders_df():
    query = '''SELECT t.bird_order, COUNT(r.id)
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        GROUP BY t.bird_order
        ORDER BY COUNT(r.id) DESC'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def families_df():
    query = '''SELECT t.bird_order, t.family, COUNT(r.id)
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        GROUP BY t.bird_order, t.family
        ORDER BY COUNT(r.id) DESC'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def geni_df():
    query = '''SELECT t.bird_order, t.family, t.genus, COUNT(r.id)
        FROM taxonomy AS t
        LEFT JOIN recordings AS r ON t.id = r.taxonomy_id
        GROUP BY t.bird_order, t.family, t.genus
        ORDER BY COUNT(r.id) DESC'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def german_species_df():
    query = '''SELECT t.bird_order, t.family, t.genus, t.species, t.german,
            COUNT(r.id), MIN(r.id), MAX(r.id)
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0
        GROUP BY t.bird_order, t.family, t.genus, t.species
        ORDER BY COUNT(r.id) DESC'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def all_recordings_df():
    query = '''SELECT t.bird_order, t.family, t.genus, t.species, t.german,
            r.id
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        ORDER BY r.id'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def german_recordings_df():
    query = '''SELECT t.bird_order, t.family, t.genus, t.species, r.id
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 AND r.downloaded IS NULL
        ORDER BY r.id'''
    df = pd.read_sql(sql=query, con=conn)
    return df


def flag_unavailable_recording(rec_id):
    query = 'UPDATE recordings SET downloaded = "NA" WHERE id = ?'
    c = conn.cursor()
    c.execute(query, (rec_id, ))
    print(f'recording {rec_id} flagged as unavailable (downloaded = 0)')
    return 1