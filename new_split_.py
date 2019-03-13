#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:30:48 2019

@author: tim
"""

from birdsong.data_preparation.balanced_split import make_split
import pandas as pd


df = pd.read_csv('label_table.csv').rename(columns={'id':'rec_id'})
df.groupby(['label', 'rec_id']).count()#.sort_values('path')

train, test = make_split(df, 20)


test.to_csv('mel_slices_test.csv')

train.to_csv('mel_slices_train.csv')
