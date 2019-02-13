#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:54:23 2019

@author: tim
"""

# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spectral_slices_prep.csv')


empties = []
for i in range(10000):
    empties.append(df.sample(64).agg({'%empty' :'mean'}).values[0])
    
plt.hist(empties, bins = 30, histtype='step')



# Total batch samples taken:
50 * ((1972 *0.8) // 64)


###############################################################################


#testing for noise instead


