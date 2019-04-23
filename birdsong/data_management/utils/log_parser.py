#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:34:37 2019

@author: tim
"""

import os
import json

log_dir = 'run_log'


records = []
for model in os.listdir(log_dir):
    for file in os.listdir(os.path.join(log_dir, model)):
        if file.endswith('.log'):
            path = os.path.join(log_dir, model, file)
            with open(path, 'r') as log:
                log_text = json.loads(log.read())
            
            records.append(log_text)
            
            
import pandas as pd

df = pd.DataFrame.from_dict(records)

top = df[df.final_accuracy_test > 0.30]