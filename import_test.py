#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:58:09 2019

@author: tim
"""

import pandas as pd
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import time
import os
from Datasets.Preprocessing.pre_preprocessing import load_audio, get_signal


###############################################################################
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
###############################################################################

# Preloader process
def preload(*args):   
    """ Expects:
    path, label, timestamps, window, stride """
    audio, sr = load_audio(path)
    signal = get_signal(audio, sr, timestamps)
    slices = slice_audio(signal, sr, window, stride)
    labels = len(slices) * [label]
    return [(slice_, label) for slice_ ,label in zip(slices, labels)]


def timeit(func):
    def wrapper(*args, **kwargs): #generic to work with any function
        t0 = time.time()
        value = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time() - t0}s')
        return value
    return wrapper

def get_entry(df, i):
    return df.path[i], df.label[i], df.timestamps[i]

def do_it(i):
    print("Proccess id: ", os.getpid())
    p, l, t = get_entry(df, i)
    result = preload(*(p, l, t, 2000, 500))
    return result



# Multiprocessing
@timeit
def multiprocess(n):
    pool = Pool(processes = n)
    results = []
    for i in range(16, 26):
        result = pool.apply_async(do_it, args = (i,))
        results.append(result)    
    output = [p.get() for p in results]
    return output

# Multithreading
@timeit
def multithread(n):
    threadpool = ThreadPool(n)
    output = threadpool.map(do_it, list(range(16, 26)))
    return output

@timeit
def sync():
    results = []
    for i in range(16, 26):
        results.append(do_it(i))
    return results


o1 = multiprocess(4)

o2 = multithread(8)

o3 = sync()



path, label, timestamps = get_entry(df, 1)

window = 1500
stride = 500

args = (path, label, timestamps)

bucket_list = [args]*10

preload(bucket_list)

def multithread(n):
    threadpool = ThreadPool(n)
    output = threadpool.map(preload, bucket_list)
    return output

o = multithread(4)


"""
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