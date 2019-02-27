import sqlite3

# Create connection to database
conn = sqlite3.connect("db.sqlite")
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE recordings
    (id INTEGER PRIMARY KEY,
    xeno_canto_id INTEGER,
    mac_aulay_id INTEGER,
    db TEXT,
    taxonomy_id INTEGER, 
    lat TEXT,
    long TEXT,
    country TEXT,
    file TEXT,
    time TEXT,
    date TEXT)''')
conn.commit()
conn.close()