from bs4 import BeautifulSoup
import requests
import json


d = {}
orders = get_orders()
for order in orders:
    print(order)
    families = get_families(order=order)
    d[order] = {}
    for family in families:
        geni = get_geni(order=order, family=family)
        d[order][family] = {}
        for genus in geni:
            species = get_species_and_subspecies(order=order, family=family, genus=genus)
            d[order][family][genus] = species

d_string = json.dumps(d)

with open('taxonomy.txt', "w+") as f:
    f.write(d_string)



def get_orders():
    url='https://www.xeno-canto.org/explore/taxonomy'
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    links = content.find_all('a')
    orders = []
    for link in links:
        orders.append(link.text)
    return orders

def get_families(order):
    url='https://www.xeno-canto.org/explore/taxonomy?o=' + order
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li')
    families = []
    for item in lists:
        if item.a.text == 'Up...' or item.a.text == order:
            pass
        else:
            try:
                families.append(item.a.text)
                #print(item.span.text)
            except:
                pass
    return families

def get_geni(order, family):
    url='https://www.xeno-canto.org/explore/taxonomy?f=' + family
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li')
    geni = []
    for item in lists:
        if item.a.text == 'Up...' or item.a.text == order or item.a.text == family:
            pass
        else:
            try: 
                geni.append(item.a.text)
                #print(item.span.text)
            except:
                pass
    return geni


def get_species_and_subspecies(order, family, genus):
    url='https://www.xeno-canto.org/explore/taxonomy?g=' + genus
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li', class_=False)
    species = {}
    for item in lists:
        if item.a.text == 'Up...' or item.a.text == order or item.a.text == family or item.a.text == genus:
            pass
        else:
            try: 
                ssps = []
                try: 
                    ssps_raw = item.ul.find_all('li')
                    for ssp in ssps_raw:
                        ssps.append(ssp.strong.text)
                except:
                    pass
                species[item.find_all('span', class_='sci-name')[0].text] = ssps
            except:
                pass
    return species