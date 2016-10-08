import requests
import json
from pprint import pprint
import os


BEER_API_KEY = os.environ['BEER_API_KEY']

response = requests.get(
	"http://api.brewerydb.com/v2/beers/",
    params={'key':BEER_API_KEY}
)
pprint(response.status_code)
pprint(json.loads(response.content.decode()))
