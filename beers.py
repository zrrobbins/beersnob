"""
download_beer_data.py

This script downloads the entire beer database from BreweryDB.
(WARNING: THIS TAKES A LONG TIME!)
"""

import requests
import json
from pprint import pprint
import os


BEER_API_KEY = os.environ['BEER_API_KEY']
BEER_DATA_DIR = "beer_data/"

if not os.path.exists(BEER_DATA_DIR):
    os.makedirs(BEER_DATA_DIR)

numberOfPages = requests.get(
	"http://api.brewerydb.com/v2/beers/",
    params={'key':BEER_API_KEY, 'p':1000000}
)

numberOfPages = json.loads(numberOfPages.content.decode())['numberOfPages']

print("Downloading {} pages of beer...".format(numberOfPages))

for page in range(1, numberOfPages+1):
	response = requests.get(
		"http://api.brewerydb.com/v2/beers/",
	    params={'key':BEER_API_KEY, 'p':page}
	)
	outfile_name = BEER_DATA_DIR + "beers_page{}.json".format(page)

	with open(outfile_name, 'w') as outfile:
		outfile.write(response.content.decode())

print("Finished!")