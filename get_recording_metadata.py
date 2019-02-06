import requests


url = 'https://search.macaulaylibrary.org/api/v1/search?mediaType=a&q=&count=100'

page = requests.get(url)

data=page.json()

print(len(data['results']['content']))