'''
This script scrapes the Xeno-Canto website for all the bird orders, families, 
geni, species and subspecies. 

The script starts here: https://www.xeno-canto.org/explore/taxonomy and walks 
every branch down till the subspecies level.

It saves it all in a jsonfile.
'''


from bs4 import BeautifulSoup
import requests
import json


def get_orders():
    """Return a list of all bird orders on the Xeno-Canto website"""
    url = 'https://www.xeno-canto.org/explore/taxonomy'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    links = content.find_all('a')
    orders = []
    for link in links:
        orders.append(link.text.lower())
    return orders


def get_families(order):
    """For a certain bird order, return all bird families on the Xeno-Canto website 
    belonging that that bird order.
    """
    url = 'https://www.xeno-canto.org/explore/taxonomy?o=' + order
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li')
    families = []
    for item in lists:
        if item.a.text == 'Up...' or item.a.text.lower() == order:
            pass
        else:
            try:
                families.append(item.a.text.lower())
            except:
                pass
    return families


def get_geni(order, family):
    """For a certain bird order and family, return all bird geni on the Xeno-Canto 
    website belonging that that bird family.
    """
    url = 'https://www.xeno-canto.org/explore/taxonomy?f=' + family
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li')
    geni = []
    for item in lists:
        if (item.a.text == 'Up...' or item.a.text.lower() == order
                or item.a.text.lower() == family):
            pass
        else:
            try:
                geni.append(item.a.text.lower())
            except:
                pass
    return geni


def get_species_and_subspecies(order, family, genus):
    """For a certain bird order, family and genus, return all bird species and 
    subspecies on the Xeno-Canto website belonging that that bird genus.
    """
    url = 'https://www.xeno-canto.org/explore/taxonomy?g=' + genus
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li', class_=False)
    species = {}
    for item in lists:
        if (item.a.text == 'Up...' or item.a.text.lower() == order
                or item.a.text.lower() == family or item.a.text.lower() == genus):
            pass
        else:
            try:
                ssps = []
                try:
                    ssps_raw = item.ul.find_all('li')
                    for ssp in ssps_raw:
                        ssps.append(ssp.strong.text.lower())
                except:
                    pass
                species[item.find_all(
                    'span', class_='sci-name')[0].text.lower()] = ssps
            except:
                pass
    return species


d = {}
orders = get_orders()
for order in orders:
    families = get_families(order=order)
    d[order] = {}
    for family in families:
        geni = get_geni(order=order, family=family)
        d[order][family] = {}
        for genus in geni:
            species = get_species_and_subspecies(
                order=order, family=family, genus=genus)
            d[order][family][genus] = species

with open('taxonomy.txt', "w+") as f:
    f.write(json.dumps(d))
