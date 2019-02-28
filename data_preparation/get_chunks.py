import pandas as pd
import sqlite3
import os

def get_records_from_classes(class_ids, seconds_per_class, min_signal_per_file=1000):
    """Get records from a list of classes. 
    Returns an amount of recordings so that the total seconds of signal
    is equal to or more than the argument seconds_per class.
    Each recording has at least min_signal_per_file milliseconds of signal.
    """
    #db_dir = os.path.join(os.getcwd(), 'storage', 'db.sqlite')
    #print(db_dir)
    #conn = sqlite3.connect(db_dir)
    if 'HOSTNAME' in os.environ:
        db_dir = '/storage/db.sqlite'
        files_dir = '/storage/german_birds/'
    else: 
        db_dir = 'storage/db.sqlite'
        files_dir = 'storage/german_birds/'
    conn = sqlite3.connect(db_dir)
    c = conn.cursor()

    result = []
    for class_id in class_ids:
        c.execute("""SELECT SUM(sum_signal) FROM recordings 
            WHERE taxonomy_id = ? AND downloaded = 1.0 AND duration < 120 AND sum_signal > ?""", 
            (class_id, min_signal_per_file / 1000))
        sum_signal = c.fetchone()[0]
        assert sum_signal, f"no recordings found for class with id {class_id} that meet the criteria"
        assert sum_signal >= seconds_per_class, f"class with id {class_id} has only {sum_signal} seconds of data that meets the requirements"
        c.execute("""SELECT r.id, t.id, r.duration, r.sum_signal, r.timestamps
            FROM recordings AS r
            JOIN taxonomy AS t
            ON r.taxonomy_id = t.id
            WHERE r.downloaded = 1.0 AND t.id = ? AND r.sum_signal >= ? AND r.duration < 120
            ORDER BY RANDOM()
            """, (class_id, min_signal_per_file / 1000))
        recordings = c.fetchall()
        cumulative_sum_signal, i = 0, 0
        while cumulative_sum_signal < seconds_per_class:
            result.append(recordings[i])
            cumulative_sum_signal += recordings[i][3]
            i += 1

    df = pd.DataFrame.from_records(result, columns=['id', 'label', 'duration', 'total_signal', 'timestamps'])
    df['path'] = df['id'].apply(lambda x: files_dir + str(x) + '.mp3')

    return df


def main():
    class_ids = [5096, 4996, 4993, 4990, 4980]
    seconds_per_class = 10
    get_records_from_classes(class_ids, seconds_per_class)


if __name__ == '__main__':
    main()
