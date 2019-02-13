import sqlite3
import urllib.request
import os

# Create connection to database
conn = sqlite3.connect("db.sqlite")
c = conn.cursor()

def get_recordings(**kwargs):
    """Get all the recordings of a specified order, family, genus or class.
    If more than one argument is given, an exception will be returned.

    The files are downloaded to the /storage directory.
    Returns a python list with all the filenames.
    """
    # check if only one argument is given:
    if len(kwargs) != 1: raise ValueError('Specify only 1 argument')
    
    # if the argument is order change it to bird_order
    if list(kwargs.keys())[0] == 'order': 
        kwargs['bird_order'] = kwargs.pop('order')

    # query database for id's
    query = ("""SELECT r.id, r.file
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.""" + list(kwargs.keys())[0] + " = '" 
            + list(kwargs.values())[0] +"'")
    c.execute(query)
    response = c.fetchall()
    
    # download all files
    downloads = []
    for recording in response:
        print(recording)
        try:
            urllib.request.urlretrieve("http:"+recording[1],
            "storage/"+str(recording[0])+".mp3")
            downloads.append(recording[0])
        except urllib.error.HTTPError:
            print(f'file {recording[0]} not found')
    
    return downloads

def get_recordings_by_species_id(species_id):
    """Get all the recordings of a specified species.

    The files are downloaded to the /storage directory.
    Returns a python list with all the filenames.
    """
    # query database for id's of the recordings
    query = """SELECT r.id, r.file
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
        WHERE t.id = ?"""
    t = (species_id,)
    c.execute(query, t)
    response = c.fetchall()
    
    # download all files
    downloads = []
    for recording in response:
        print(recording)
        try:
            urllib.request.urlretrieve("http:"+recording[1],
            "storage/german_birds/"+str(recording[0])+".mp3")
            print(f'file {recording[0]} downloaded')
            downloads.append(recording[0])
        except urllib.error.HTTPError:
            print(f'file {recording[0]} not found')
    
    return downloads


def get_recording_by_id(recording_id):
    """Get a specified recording.

    The file is downloaded to the /storage directory.
    Returns 1 if succesful, 0 if failed.
    """
    try:
        urllib.request.urlretrieve("http://www.xeno-canto.org/" 
            + str(recording_id) + "/download",
        "storage/german_birds/" + str(recording_id) + ".mp3")
        print(f'file {recording_id} downloaded')
        return 1
    except urllib.error.HTTPError:
        print(f'file {recording_id} not found')
        return 0