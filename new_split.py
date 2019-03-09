#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:06:53 2019

@author: tim
"""
import pandas as pd
import numpy as np
import itertools
df_train = pd.read_csv('storage/df_train_local.csv')
df_test = pd.read_csv('storage/df_test_local.csv')
#label_codes = pd.read_csv('storage/label_codes.csv')


#df = pd.concat([df_train, df_test])
#df = df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])

CLASSES = 10

counts = df.groupby(['label', 'rec_id']).path.count()

"""
So I want rec_ids from each class so that the sum of slices for each class is as 
close as possible to 200
"""
counts = counts.reset_index()
counts.sort_values(['label', 'path'], inplace = True)
counts.reset_index(drop = True, inplace = True)


def get_combinations(df, cl, value):
    """ for one class label and desired total value of slices, compute
    all possible permutations of 2 - 4 recordings (choice based on average
    number of recordings per class) and compute the sum of slices for each
    combination of recordings. From the 10 combinations closest to the
    desired value, randomly pick one and return the rec_ids for those.
    """
    subset = df[(df.label == cl)].reset_index(drop = True).reset_index(drop = True)
    
    sums = []
    indices = []
    for j in range(2, 5):
        permutations  = itertools.permutations( range(len(subset)), j )
        for tup in permutations:
            sums.append( sum([subset.path[i] for i in tup]) )
            indices.append(tup)
            
            
    idx = np.vstack([np.array(indices), (np.abs(np.array(sums)-value)) ])
    
    random_best_idx = np.random.choice(np.flatnonzero(idx[1] == idx[1].min()))
    
    random_best = idx[0, random_best_idx]
    
    return subset.rec_id[list(random_best)].values

def get_rec_ids(df, value):
    rec_ids = []
    for i in range(CLASSES):
        rec_ids.append(get_combinations(df, i, value))    
    return np.hstack(rec_ids)

rec_ids = get_rec_ids(counts, 200)

test_df = df[df.rec_id.isin(rec_ids)]


test_df.groupby('label').count()

train_df = df[~df.rec_id.isin(rec_ids)]

train_df.groupby('label').count()


test_df.to_csv('storage/df_test_local.csv')
train_df.to_csv('storage/df_train_local.csv')