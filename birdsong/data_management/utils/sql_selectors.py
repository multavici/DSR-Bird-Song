import os
import pandas as pd

def lookup_species_by_rec_id(c, rec_id):
    """ For a cursor and a recording id, look up the foreground species for 
    this recording and return label as 'genus_species' """
    c.execute("""
        SELECT r.taxonomy_id, t.genus, t.species FROM recordings as r 
        JOIN taxonomy as t ON r.taxonomy_id = t.id
        WHERE r.id = ?
        """, (rec_id,))
    fetch = c.fetchone()
    return fetch[1] + "_" + fetch[2]

def lookup_recordings_to_download(c, label, nr_recordings):
    """ For a cursor and label ('genus_species'), return nr_recordings
    if bird is german and recordings have not been downloaded yet """
    genus, species = label.split('_')
    c.execute("""
        SELECT r.id, r.file
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded IS NULL 
        AND t.genus = ?
        AND t.species = ?
        LIMIT ? """, (genus, species, nr_recordings))
    recordings = c.fetchall()
    return list(map((lambda x: (x[0],'http:' + x[1])), recordings))


def lookup_downloaded_german_recordings(conn):
    """ Check how many recordings of german birds have already been downloaded """
    query = """ 
        SELECT r.id, r.file, r.scraped_duration, (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded = 1.0
        AND r.scraped_duration IS NOT NULL """
    return pd.read_sql(query, conn)

def lookup_not_downloaded_german_recordings(conn):
    """ Check how many recordings of german birds have not been downloaded yet """
    query = """ 
        SELECT r.id, r.file, r.scraped_duration, (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded IS NULL 
        AND r.scraped_duration > 5.0 """                                     #TODO: Set these with environment variables
    return pd.read_sql(query, conn)
    
def lookup_all_recordings(conn, n_seconds=5.0):
    """ Return a df of all recordings that are
    longer than the window size of the spectrogram function """                 #TODO: Set these with environment variables
    query = """ 
        SELECT r.id, r.file, r.sum_signal, r.scraped_duration, (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.scraped_duration > ? """
    return pd.read_sql(query, conn, params=(n_seconds,))
    

def lookup_recordings_for_noise(c, label, nr_recordings):
    """ Return random recordings for soundscape noise bank """
    genus, species = label.split('_')
    c.execute("""
        SELECT r.id, r.file
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND t.genus = ?
        AND t.species = ?
        ORDER BY RANDOM()
        LIMIT ? """, (genus, species, nr_recordings))
    recordings = c.fetchall()
    return list(map((lambda x: (x[0],'http:' + x[1])), recordings))

# Used to set files to 'downloaded' if in rec_id_list
def set_downloaded(c, id_list):
    if len(id_list) == 1:
        c.execute('UPDATE recordings SET downloaded = 1.0 WHERE id = ' + str(id_list[0]))
    else:
        c.execute('UPDATE recordings SET downloaded = 1.0 WHERE id IN ' +
                  str(tuple(id_list)))
