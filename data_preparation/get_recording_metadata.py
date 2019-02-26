import requests
import pandas as pd


# Xeno-Canto library
birds = pd.read_csv('german_birds_in_db')

for index, bird in birds.iterrows():
    print(bird['genus']+'%20'+bird['species'])
    ans_dict = get_bird_songs(bird['genus']+'%20'+bird['species'])
    print(ans_dict['numSpecies'])
    print(len(ans_dict['recordings']))






# MacAulay Library
url = 'https://search.macaulaylibrary.org/api/v1/search?mediaType=a&q=&count=100'

page = requests.get(url)

data=page.json()

print(len(data['results']['content']))