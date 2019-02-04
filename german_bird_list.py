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

for i, table in enumerate(ger_birds):
    if table.iloc[0,:].isna().all():
        family = table.iloc[1,0].split(' - ')[0]
        # Get rid of image column
        table.drop(table.columns[0], axis = 1, inplace = True)
        table.columns = ['german_name', 'latin_name', 'status', 'comment']
        table['family'] = family
        table.drop(table.index[:3], inplace=True)    
    else:
        print(f'Issue with first line in table {i}')
    

def request_url(URL):
    res = requests.get(URL)
    res.raise_for_status()
    return res

r = request_url(url)
soup = bs4.BeautifulSoup(r.text, 'html.parser') 
elems = soup.select('h2')
orders = [elem.text.split(' - ')[0] for elem in elems][1:-2]




# Nr. of all birds
sum([len(x) for x in ger_birds])

