import sqlite3
import urllib.request
import pandas as pd
import os

# Check if script is run locally or on server
if 'HOSTNAME' in os.environ:
    # script runs on server
    DATABASE_DIR = '/storage/db.sqlite'
else:
    # script runs locally
    DATABASE_DIR = 'storage/db.sqlite'

conn = sqlite3.connect(DATABASE_DIR)
c = conn.cursor()


def get_recordings_info():
    query = '''SELECT t.id AS taxonomy_id, t.bird_order, t.genus, t.family, t.species, t.german, 
            r.id, r.xeno_canto_id, r.lat, r.long, r.country, r.file, r.time, r.date, 
            r.duration, r.sum_signal, r.timestamps, r.downloaded, r.q
        FROM taxonomy AS t
        JOIN recordings AS r ON t.id = r.taxonomy_id
    '''
    return pd.read_sql(query, con=conn)


def get_downloaded_records_from_classes(class_ids, base_path,
                                        seconds_per_class=None, min_signal_per_file=5000):
    """Get records from a list of classes. 

    If seconds_per_class is not defined the function returns all downloaded records from that class.

    If seconds_per_class is defined the function returns the least amount of recordings so that the total seconds of signal
    is equal to or more than seconds_per_class.

    Each recording has at least min_signal_per_file milliseconds of signal.
    """
    result = []
    for class_id in class_ids:
        if seconds_per_class:
            c.execute("""SELECT SUM(sum_signal) FROM recordings 
                WHERE taxonomy_id = ? AND downloaded = 1.0 AND sum_signal > ? AND duration < 500""",
                      (class_id, min_signal_per_file / 1000))
            sum_signal = c.fetchone()[0]
            assert sum_signal, f"no recordings found for class with id {class_id} that meet the criteria"
            assert sum_signal >= seconds_per_class, f"class with id {class_id} has only {sum_signal} seconds of data that meets the requirements"
        c.execute("""SELECT r.id, r.xeno_canto_id, t.id, r.duration, r.sum_signal, r.timestamps
            FROM recordings AS r
            JOIN taxonomy AS t
            ON r.taxonomy_id = t.id
            WHERE r.downloaded = 1.0 AND t.id = ? AND r.sum_signal >= ? AND duration < 500""",
                  (class_id, min_signal_per_file / 1000))
        recordings = c.fetchall()
        if seconds_per_class:
            cumulative_sum_signal, i = 0, 0
            while cumulative_sum_signal < seconds_per_class:
                result.append(recordings[i])
                cumulative_sum_signal += recordings[i][4]
                i += 1
        else:
            for recording in recordings:
                result.append(recording)

    df = pd.DataFrame.from_records(result, columns=[
                                   'id', 'xeno-canto_id', 'label', 'duration', 'total_signal', 'timestamps'])
    df['path'] = df['id'].apply(lambda x: base_path + str(x) + '.mp3')

    return df


if __name__ == '__main__':
    print(get_recordings_info())
