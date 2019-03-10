'''
This script creates the taxonomy table in the sqlite database. And then fills it
up with the scraped data in taxonomy.txt
'''

import sqlite3
import json
import pandas as pd
import os

if 'HOSTNAME' in os.environ:
    # script runs on server
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    DATABASE_DIR = 'storage/db.sqlite'

# Create connection to database
conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()


def create_taxonomy():
    conn.execute('''CREATE TABLE taxonomy(
        id INTEGER PRIMARY KEY, 
        bird_order TEXT, 
        family TEXT, 
        genus TEXT, 
        species TEXT, 
        german BOOLEAN, 
        xeno_canto BOOLEAN, 
        mc_aulay BOOLEAN)''')
    # order is a reserved key-word so it's called bird_order
    conn.commit()


def add_xeno_canto_species():
    # Read JSON text file
    with open('taxonomy.txt') as f:
        tax_string = f.read()
    tax_dict = json.loads(tax_string)

    # Save to database table
    for order, families in tax_dict.items():
        for family, geni in families.items():
            for genus, species in geni.items():
                for specie in species:
                    conn.execute("""
                        INSERT INTO taxonomy (bird_order, family, genus, 
                            species, xeno_canto) 
                        VALUES (?, ?, ?, ?, ?)
                        """,
                                 (order, family, genus, specie.split(" ")[1], True))
                # Use this to also save subspecies in database
                '''for specie, subspecies in species.items():
                    for subscspecie in subspecies:
                        conn.execute("""
                        INSERT INTO taxonomy (bird_order, family, genus, 
                            species, subspecies) 
                        VALUES (?, ?, ?, ?, ?)
                        """, (order, family, genus, specie, subscspecie)  )'''
    conn.commit()


def add_german_species():
    birds = pd.read_csv('german_birds.csv')

    for index, bird in birds.iterrows():
        genus = bird['genus']
        species = bird['species']

        c.execute('''UPDATE taxonomy SET german = 1 
            WHERE species = ? and genus = ?''', (species, genus))

        # If update was not succesful, add the species
        if c.rowcount != 1:
            order = bird['order']
            family = bird['family']
            c.execute('''INSERT INTO taxonomy (bird_order, family, genus, 
                species, german) 
                VALUES (?, ?, ?, ?, 1)''', (order, family, genus, species))

    conn.commit()


create_taxonomy()
add_xeno_canto_species()
add_german_species()
conn.close()
