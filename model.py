#%%

import sqlite3
import json
import pandas as pd

# Create connection to database
conn = sqlite3.connect("db.sqlite")

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

print(geni_df())