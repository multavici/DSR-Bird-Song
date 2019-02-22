#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:58:09 2019

@author: tim
"""

from Datasets.dynamic_dataset import preload
import pandas as pd


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

def get_entry(df, i):
    return df.path[i], df.label[i], df.timestamps[i]

p, l, t = get_entry(df, 1)

sample = preload(p,l,t, 300, 100)