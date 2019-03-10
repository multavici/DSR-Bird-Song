
# Test Run
import sqlite3
import pandas as pd
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

from torch.utils.data import DataLoader
from Datasets.static_dataset import SpectralDataset
from models.bulbul import Bulbul
import os

##########################################################################
# Get df of paths for pickled slices
def get_df(): 
    conn = sqlite3.connect('storage/db.sqlite')
    c = conn.cursor()
    def lookup(id):
        c.execute("""SELECT r.taxonomy_id, t.genus, t.species FROM recordings as r 
            JOIN taxonomy as t 
                ON r.taxonomy_id = t.id
            WHERE r.id = ?""", (id,))
        fetch = c.fetchone()
        return fetch[1] + "_" + fetch[2]
    list_recs = []
    for dirpath, dirname, filenames in os.walk('storage/slices'):
        for name in filenames:
            path = os.path.join(dirpath, name)
            id = dirpath.split("/")[2]
            species = lookup(id)
            list_recs.append((str(path), species))   
    df = pd.DataFrame(list_recs, columns=['path', 'label'])
    return df

df = get_df()
print(df.groupby('label').count())
df.to_csv('slices_and_labels.csv')