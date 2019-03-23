#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:06:53 2019

@author: tim
"""
import pandas as pd
import numpy as np
import itertools

def get_combinations(df, class_, value):
    subset = df[(df.label == class_) & (df.path <= value)].reset_index(drop = True)

    def _pick(stored, count, subset, tolerance):
        """ Pick random recording if > value slices pick new, if value, return, else:
        keep it pick new random one, if sum > value, pick new, if value return, else:
        continue """
        random_choice = subset.sample(n=1)
        if random_choice.path.values + count > value + tolerance:
            stored, count = _pick(stored, count, subset, tolerance)
            return stored, count
        if random_choice.path.values + count < value - tolerance:
            subset = subset.drop(random_choice.index)
            count += random_choice.path.values
            stored.append(int(random_choice.rec_id.values))
            stored, count = _pick(stored, count, subset, tolerance)
            return stored, count
        else:
            stored.append(int(random_choice.rec_id.values))
            count += random_choice.path.values
            return stored, count

    def _call(tolerance):
        try:
            ids, count = _pick([], 0, subset, tolerance)
            return ids, count
        except:
            ids, count = _call(tolerance + 1)
            return ids, count

    ids, count = _call(0)
    return np.array(ids)

def make_split(df, test_samples=200, train_samples=2200, both_even=False):
    """ Requires a Dataframe with label, rec_id, path as columns for respective
    slices.
    The script is grouping this Dataframe by label and rec_id and will count the
    number of slices for each. It is then going through possible combinations
    of recordings for each class that result in a desired number of unique slices
    for each class in a test set.
    If needed, one can also pass a desired number of unique slices per class for
    the training set which is then randomly up or downsampled if both_even is
    True.
    """
    df['rec_id'] = df.path.apply(lambda x: int(x.split('_')[0]))
    
    def get_rec_ids(df, value):
        rec_ids = []
        for class_ in classes:
            print(f'Splitting records for {class_}')
            rec_ids.append(get_combinations(df, class_, value))
        return np.hstack(rec_ids)

    # Unique class names
    classes = set(df.label)

    # Aggregating counts for rec_ids
    counts = df.groupby(['label', 'rec_id']).path.count().reset_index()
    counts.sort_values(['label', 'path'], inplace = True)
    counts.reset_index(drop = True, inplace = True)
    min_possible = counts.groupby('label').agg({'label':'min', 'path':'min'}).max()

    if int(min_possible[1]) > test_samples:
        print(f"Cant make an even split of {test_samples}: For class '{min_possible[0]}' the recording with the least number of slices already has {min_possible[1]}")

    # Getting combinations of rec_ids whose slices sum up to the desired value
    rec_ids = get_rec_ids(counts, test_samples)

    # Get test dataset
    test_df = df[df.rec_id.isin(rec_ids)].reset_index(drop = True)
    #print(f"Class split for test_df: \n {test_df.groupby('label').count()} \n")

    # Get remaining
    rest_train_df = df[~df.rec_id.isin(rec_ids)]

    # If desired return even dataset for train as well
    if both_even:
        train_df = rest_train_df.groupby('label').apply(lambda x: x.sample(n = train_samples)).reset_index(drop = True)
        #print(f"Class split for train_df: \n {train_df.groupby('label').count()}")
        return train_df, test_df

    else:
        #print(f"Class split for train_df: \n {rest_train_df.groupby('label').count()}")
        return rest_train_df, test_df
