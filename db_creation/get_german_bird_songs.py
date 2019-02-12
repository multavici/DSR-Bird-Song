from get_song import get_bird_songs
import pandas as pd
import json

birds = pd.read_csv('german_birds_in_db')

#print(birds)

#print(len(get_bird_songs(birds['genus'][0]+' '+birds['species'][0])))

'''
for index, bird in birds.iterrows():
    print(bird['genus']+'%20'+bird['species'])
    ans_dict = get_bird_songs(bird['genus']+'%20'+bird['species'])
    print(ans_dict['numSpecies'])
    print(len(ans_dict['recordings']))
'''

import urllib.request

sample = ['oxyura%20jamaicensis', 
    'cygnus%20columbianus', 
    'anser%20fabalis', 
    'anser%20brachyrhynchus']


def get_recordings(species):
    for specie in species:
        ans_dict = get_bird_songs(specie)
        recordings = ans_dict['recordings']
        i = 0
        for recording in recordings:
            print(recording['file'])
            urllib.request.urlretrieve ("http:"+recording['file'], "songs/"+genus+str(i)+".mp3")
            i += 1

