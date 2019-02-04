from bs4 import BeautifulSoup
import requests
import json

geni_dict = {}

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
                print(item.span.text)
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
                print(geni.span.text)
            except:
                pass
    return geni


def get_species(order, family, genus):
    url='https://www.xeno-canto.org/explore/taxonomy?f=' + family
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    content = soup.find(id='content-area')
    lists = content.find_all('li')
    species = []
    for item in lists:
        if item.a.text == 'Up...' or item.a.text == order or item.a.text == family, item.a.text == :
            pass
        else:
            try: 
                geni.append(item.a.text)
                print(geni.span.text)
            except:
                pass
    return geni


def get_bird_songs(genus=None):
    url = "https://www.xeno-canto.org/api/2/recordings?query="+genus
    page = requests.get(url)
    data = page.json()
    print(data)
    #soup = BeautifulSoup(page.text, 'html.parser')
    #recordings_dict = json.loads(soup)
    #print(recordings_dict['numRecordings'])


#get_bird_songs(genus='Struthio')

orders = get_orders()
print(orders)
d = {}
for order in orders:
    families = get_families(order=order)
    d[order] = {}
    for family in families:
        geni = get_geni(order=order, family=family)
        d[order][family] = {}



print(d)



print(d)
