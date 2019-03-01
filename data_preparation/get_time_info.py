"""
This script runs the signal_timestamps function, that gets the duration of each 
recording + all the timestamps of the recoring, and saves the info in the 
database.
"""

import os
from Signal_Extraction import signal_timestamps
import sqlite3


downloaded_files = os.listdir("storage/top10_german_birds")
print('list with downloaded files made')
print(len(downloaded_files))
downloaded_ids = [int(x[:-4]) for x in downloaded_files]
conn = sqlite3.connect("storage/db.sqlite")
print('database loaded')
c = conn.cursor()
c.execute("""select id from recordings where downloaded = 1.0""")
processed_ids = [i[0] for i in c.fetchall()]
print('list of already processed recordings')
print(len(processed_ids))

to_process = [x for x in downloaded_ids if x not in processed_ids]
print('list of files to process')
print(len(to_process))

batch = []
for i, rec_id in enumerate(to_process):
    rec = str(rec_id) + '.mp3'
    print(rec_id)
    try:
        duration, sum_signal, timestamps = signal_timestamps(
            "storage/top10_german_birds/" + rec)
        batch.append((duration, sum_signal, timestamps, rec_id))
        if len(batch) % 10 == 0:
            print(f"batch {i} full")
            c.executemany("""UPDATE recordings SET downloaded = 1.0, duration = ?, sum_signal = ?, timestamps = ? 
                WHERE id = ?""", batch)
            conn.commit()
            batch = []
    except:
        pass
