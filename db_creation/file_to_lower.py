from itertools import chain
from glob import glob

file = open('taxonomy.txt', 'r')

lines = [line.lower() for line in file]
with open('taxonomy.txt', 'w') as out:
     out.writelines(sorted(lines))