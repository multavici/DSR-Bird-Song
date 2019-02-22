# get german recordings
import pandas as pd
from model import german_recordings_df
german_recordings = german_recordings_df()

# get already downloaded files
import os
downloaded = os.listdir("/storage/german_birds")
downloaded_ids = []
for file in downloaded:
    try:
        downloaded_ids.append(int(file[:-4]))
    except:
        pass
print(len(downloaded))

# remove already downloaded files from the list
downloaded = german_recordings['id'].isin(downloaded_ids)
to_download = german_recordings.loc[~downloaded,:]

# download recordings
import urllib.request
for id_rec in to_download['id']:
    try:
        urllib.request.urlretrieve("http://www.xeno-canto.org/" 
            + str(id_rec) + "/download",
            "/storage/german_birds/" + str(id_rec) + ".mp3")
        print(f'file {id_rec} downloaded')
    except urllib.error.HTTPError:
        print(f'file {id_rec} not found')
        pass

