import os
import pandas as pd

def species_by_rec_id(conn, rec_id):
    """ For a cursor and a recording id, look up the foreground species for 
    this recording and return label as 'genus_species' """
    c = conn.cursor()
    c.execute("""
        SELECT r.taxonomy_id, t.genus, t.species FROM recordings as r 
        JOIN taxonomy as t ON r.taxonomy_id = t.id
        WHERE r.id = ?
        """, (rec_id,))
    fetch = c.fetchone()
    return fetch[1] + "_" + fetch[2]

def recordings_to_download(conn, label, desired_slices):
    """ For a cursor, a species label ('genus_species') and a desired number
    of slices, return a list of tuples of rec_ids and urls for a number of 
    recordings that are expected to suffice that number of slices. """
    genus, species = label.split('_')
    
    needed_signal = 5 + (desired_slices * 2.5) #TODO: Set with environment variables
    needed_duration = needed_signal * 2 #Safety factor, based on experiments
    
    query = f"""
        SELECT r.id as rec_id, ('http:' || r.file) as url, r.scraped_duration
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded IS NULL 
        AND t.genus = '{genus}'
        AND t.species = '{species}' 
        AND r.scraped_duration < 1000.0 """
    
    recordings = pd.read_sql(query, conn)
    recordings['cumulative'] = recordings.scraped_duration.cumsum()
    
    # The first time the cumsum of duration surpasses the needed duration
    if recordings.cumulative.max() >= needed_duration: 
        index = recordings[recordings.cumulative > needed_duration].cumulative.idxmin()
    else:
        index = recordings.cumulative.idxmax()
        print(f'For {label} we would need ~{needed_duration} but only {recordings.cumulative.max():.1f} seconds remain.')
    needed_recordings = recordings.iloc[:index+1]
    return list(needed_recordings[['rec_id', 'url']].itertuples(index=False, name=False))

def downloaded_german_recordings(conn):
    """ Check how many recordings of german birds have already been downloaded """
    query = """ 
        SELECT r.id, r.file, r.scraped_duration, (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded = 1.0
        AND r.scraped_duration IS NOT NULL """
    return pd.read_sql(query, conn)

def not_downloaded_german_recordings(conn):
    """ Check how many recordings of german birds have not been downloaded yet """
    query = """ 
        SELECT r.id, r.file, r.scraped_duration, (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded IS NULL 
        AND r.scraped_duration > 5.0 """                                     #TODO: Set these with environment variables
    return pd.read_sql(query, conn)
    
def duration_per_not_downloaded_german_species(conn, selection):
    tuple = *list(selection.label),
    query = f""" 
        SELECT (t.genus || '_' || t.species) AS label, sum(r.scraped_duration) as total_audio
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.downloaded IS NULL 
        AND r.scraped_duration > 5.0
        AND r.scraped_duration < 1000.0  
        AND label IN {tuple}                             
        GROUP BY label  
        ORDER By total_audio DESC """  
    return pd.read_sql(query, conn)
    
def top_k_duration_all_recordings(conn, k):
    """ Return a df of all recordings that are
    longer than the window size of the spectrogram function """                 #TODO: Set these with environment variables
    query = """ 
        SELECT (t.genus || '_' || t.species) AS label
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.german = 1.0 
        AND r.scraped_duration > 5.0                                  
        GROUP BY label  
        ORDER BY sum(r.scraped_duration) DESC
        LIMIT ? """  
    return pd.read_sql(query, conn, params=(k,))
    
def recordings_for_noise(conn, label, nr_recordings):
    """ Return random recordings for soundscape noise bank """
    genus, species = label.split('_')
    c = conn.cursor()
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

def reset_downloaded(conn):
    c = conn.cursor()
    c.execute('UPDATE recordings SET downloaded = NULL')
    conn.commit()
    
# Used to set files to 'downloaded' if in rec_id_list
def set_downloaded(conn, id_list):
    c = conn.cursor()
    if len(id_list) == 1:
        c.execute('UPDATE recordings SET downloaded = 1.0 WHERE id = ' + str(id_list[0]))
    else:
        c.execute('UPDATE recordings SET downloaded = 1.0 WHERE id IN ' +
                  str(tuple(id_list)))
    conn.commit()
