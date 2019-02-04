#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:14:16 2019

@author: tim
"""

import pandas as pd
import bs4, requests

url = 'https://commons.wikimedia.org/wiki/Liste_der_V%C3%B6gel_Deutschlands'
ger_birds = pd.read_html(url)

def request_url(URL):
    res = requests.get(URL)
    res.raise_for_status()
    return res

def clean_table():
    error_list = []
    error_count = 0
    for i, table in enumerate(ger_birds):
        
        #Ensure formatting:
        assert table.iloc[0,:].isna().all(), f'Issue with first line in table {i}'
        
        #Retrieve family
        family = table.iloc[1,0].split(' - ')[0]
        if family == 'Bild':
            table.drop(table.index[0], axis = 0, inplace = True)
            family = table.iloc[1,0].split(' - ')[0]
            
        # Get rid of image column
        table.drop(table.columns[0], axis = 1, inplace = True)
        table.columns = ['german_name', 'latin_name', 'status', 'comment']
        table['family'] = family
        table.drop(table.index[:3], inplace=True)    
        
        # Get order name 
        try:
            print(f'Retrieving order for {family}')
            test = request_url(f'https://paleobiodb.org/data1.2/taxa/list.json?name={family}&rel=parent')
            order = test.json()['records'][0]['nam']
            table['order'] = order
        except:
            print(f'Error in retreiving order for {family}')
            error_count += 1
            error_list.append(family)
            table['order'] = 'NaN'
            
    print(f'{error_count} failed order lookups')
    return error_list

errors = clean_table()

# Nr. of all birds
#sum([len(x) for x in ger_birds])

