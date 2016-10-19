# beersnob
Your favorite neighborhood machine-learning beer snob.


## Purpose
This program is our final project for our Artificial Intelligence class at WPI. We aimed to see if it was possible to classify beers by style (e.g. IPA, Pale Ale) based on different sets of attributes of that beer (e.g. abv, ibu, etc.). This project makes use of the [scikit-learn library](www.scikit-learn.org) for our classifiers, and [BreweryDB](www.brewerydb.com) for our beer data.


## Setup
This project uses Python 3.5. After cloning/forking the repo on to your machine, install requirements using pip in a new virtualenv or directly on to your machine:
`pip install -r requirements.txt`

After the environment is setup, you must download the beer data. You will need a premium API key from BreweryDB to continue. Export your API key as BEER_API_KEY to an environment variable. Then run download_beer_data.py (note this could take a long time, there is a lot of data!):
`python download_beer_data.py`

Now you need to transform the json data into a format usable by the classifiers. To do this, run data_transformer.py:
`python data_transformer.py`


## Run
To run the program, simply call `python main.py [trim_level]`, where `trim_level` is optional. You may enable cross-validation, change the attributes for the classifiers to learn with (trim_level), and change the amount of train/test data via variables in the main.py file.
