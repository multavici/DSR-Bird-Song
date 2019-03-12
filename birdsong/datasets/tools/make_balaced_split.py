#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:06:53 2019

@author: tim
"""
import pandas as pd
import numpy as np
import itertools
df = pd.read_csv('slices_and_labels3.csv')




df.groupby('label').count()



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
    for j in range(3, 5):
        combinations  = itertools.combinations( range(len(subset)), j )
        for tup in combinations:
            sums.append( sum([subset.path[i] for i in tup]) )
            indices.append(tup)
            
            
    idx = np.vstack([np.array(indices), (np.abs(np.array(sums)-value)) ])
    
    random_best_idx = np.random.choice(np.flatnonzero(idx[1] == idx[1].min()))
    
    random_best = idx[0, random_best_idx]
    
    return subset.rec_id[list(random_best)].values

def make_split(df, test_samples=200, train_samples=2200, both_even=False):
    """
    So I want rec_ids from each class so that the sum of slices for each class is as 
    close as possible to value
    """
    def get_rec_ids(df, value):
        rec_ids = []
        for class_ in classes:
            print(i)
            rec_ids.append(get_combinations(df, class_, value))    
        return np.hstack(rec_ids)

    # Unique class names
    classes = set(df.label)

    # Aggregating counts for rec_ids
    counts = df.groupby(['label', 'rec_id']).path.count().reset_index()
    counts.sort_values(['label', 'path'], inplace = True)
    counts.reset_index(drop = True, inplace = True)

    # Getting combinations of rec_ids whose slices sum up to the desired value
    rec_ids = get_rec_ids(counts, test_samples)

    # Get test dataset
    test_df = df[df.rec_id.isin(rec_ids)].reset_index(drop = True)
    print(f'Class split for test_df: \n {test_df.groupby('label').count()}')
    
    # Get remaining
    rest_train_df = df[~df.rec_id.isin(rec_ids)].drop(columns = 'Unnamed: 0')

    # If desired return even dataset for train as well
    if both_even:
        train_df = rest_train_df.groupby('label').apply(lambda x: x.sample(n = train_samples)).reset_index(drop = True)
        print(f'Class split for train_df: \n {train_df.groupby('label').count()}')
        return train_df, test_df
    
    else:
        print(f'Class split for train_df: \n {rest_train_df.groupby('label').count()}')
        return rest_train_df, test_df
