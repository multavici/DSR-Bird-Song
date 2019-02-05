import sqlite3
import json

conn = sqlite3.connect("bird_song.sqlite")

conn.execute('''CREATE TABLE taxonomy
    (id INTEGER PRIMARY KEY, bird_order TEXT, family TEXT, genus TEXT, species TEXT, subspecies TEXT)''') # order is a reserved key-word so it's called bird_order

conn.commit()

with open('taxonomy.txt') as f:
    tax_string = f.read()

tax_dict = json.loads(tax_string)

print(type(tax_dict))

for order, families in tax_dict.items():
    for family, geni in families.items():
        for genus, species in geni.items():
            for specie, subspecies in species.items():
                for subscspecie in subspecies:
                    print(order)
                    print(family)
                    print(genus)
                    print(specie)
                    print(subscspecie)
                    print((order, family, genus, specie, subscspecie))
                    conn.execute("""
                    INSERT INTO taxonomy (bird_order, family, genus, species, subspecies) VALUES (?, ?, ?, ?, ?)
                    """, (order, family, genus, specie, subscspecie)  )

conn.commit()