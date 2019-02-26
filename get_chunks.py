import pandas as pd
import sqlite3
import os


def get_records_from_classes(class_ids, seconds_per_class):
    """Get records from a list of classes. 
    Returns an amount of recordings so that the total seconds of signal
    is equal to or more than the argument seconds_per class
    """
    #db_dir = os.path.join(os.getcwd(), 'storage', 'db.sqlite')
    #print(db_dir)
    #conn = sqlite3.connect(db_dir)
    conn = sqlite3.connect('/storage/db.sqlite')
    c = conn.cursor()

    result = []
    for class_id in class_ids:
        c.execute(
            "SELECT SUM(sum_signal) FROM recordings WHERE taxonomy_id = ?", (class_id,))
        sum_signal = c.fetchone()[0]
        print(sum_signal)
        assert sum_signal >= seconds_per_class, f"class with id {class_id} has only {sum_signal} seconds of data"
        c.execute("""SELECT r.id, t.id, r.duration, r.sum_signal, r.timestamps
            FROM recordings AS r
            JOIN taxonomy AS t
            ON r.taxonomy_id = t.id
            WHERE r.downloaded = 1.0 AND t.id = ?
            ORDER BY RANDOM()
            """, (class_id, ))
        recordings = c.fetchall()
        cumulative_sum_signal, i = 0, 0
        while True:
            result.append(recordings[i])
            cumulative_sum_signal += recordings[i][3]
            print(cumulative_sum_signal)
            if cumulative_sum_signal > seconds_per_class:
                break
            i += 1

    df = pd.DataFrame.from_records(result, columns=['id', 'label', 'duration', 'total_signal', 'timestamps'])
    df['path'] = df['id'].apply(lambda x: os.path.join(os.getcwd(), 'storage', 'german_birds', str(x) + '.mp3'))
    return df


def main():
    class_ids = [5096, 4996, 4993, 4990, 4980]
    seconds_per_class = 10
    get_records_from_classes(class_ids, seconds_per_class)


if __name__ == '__main__':
    main()
