#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:58:09 2019

@author: tim
"""

from Datasets.dynamic_dataset import preload
import pandas as pd
from multiprocessing import Pool

from multiprocessing.pool import ThreadPool

df = pd.read_csv('Testing/test_df.csv')
def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col
df.label = label_encoder(df.label)
# Remove that one file without signal
df = df[df.total_signal > 0].reset_index(drop = True)


def get_entry(df, i):
    return df.path[i], df.label[i], df.timestamps[i]


pool = Pool(processes = 3)

results = []
for i in range(16, 26):
    p,l,t = get_entry(df, i)
    result = pool.apply_async(preload, args = (p ,l , t, 1500, 500))
    results.append(result)
    
output = [p.get() for p in results]


#Multithreading instead
"""
def threads(urls):
    pool = ThreadPool(6)
    results = pool.map(requests.get, urls)
    return results
"""

threadpool = ThreadPool(6)


for i in range(16, 26):
    p,l,t = get_entry(df, i)

threadpool.map(preload, )




"""
sr = 22050


###############################################################################
# Comparing actual and computed length
def check_length(df, i, window, stride):
    signal_s = df.total_signal[i]
    computed_nr_samples = ((signal_s - (window/1000)) // (stride/1000)) + 1 
    p, l, t = get_entry(df, i)
    actual_nr_samples = len(preload(p,l,t, window, stride))

    return computed_nr_samples, actual_nr_samples
        
check_length(df, 4, 500, 200)

coll = []
for i in range(len(df)):
    print(i)
    coll.append(check_length(df, i, 500, 200))


import numpy as np

a = np.array(coll)

# One error -1 for 1 out of 159 sound files:
# Make correction when defining dataset length
"""