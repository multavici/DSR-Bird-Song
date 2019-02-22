#%%

import sqlite3
import json
import pandas as pd

# Create connection to database
conn = sqlite3.connect("../storage/db.sqlite")

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
        WHERE t.german = 1.0
        ORDER BY r.id'''
    df = pd.read_sql(sql=query, con=conn)
    return df