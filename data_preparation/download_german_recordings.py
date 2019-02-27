# get german recordings
import pandas as pd
from model import german_recordings_df, flag_unavailable_recording
import urllib.request
import os

german_recordings = german_recordings_df()
print(f'german recordings: {len(german_recordings.index)}')

# get already downloaded files
files_dir = os.path.join(os.getcwd(), 'storage', 'german_birds')
downloaded = os.listdir(files_dir)
downloaded_ids = []
for file in downloaded:
    try:
        downloaded_ids.append(int(file[:-4]))
    except:
        pass
print(f'files downloaded already: {len(downloaded)}')

# remove already downloaded files from the list
downloaded = german_recordings['id'].isin(downloaded_ids)
to_download = german_recordings.loc[~downloaded,:]
print(f'files still to download: {len(to_download.index)}')

# download recordings
for id_rec in to_download['id']:
    try:
        urllib.request.urlretrieve("http://www.xeno-canto.org/" 
            + str(id_rec) + "/download",
            os.path.join(files_dir, str(id_rec) + ".mp3"))
        print(f'file {id_rec} downloaded')
    except urllib.error.HTTPError:
        print(f'file {id_rec} not found')
        flag_unavailable_recording(id_rec)
        pass

