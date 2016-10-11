"""
data_transformer.py

File for manipulating downloaded API data into correct format for use as a scikit-learn dataset.

"""
import json
import os
from pprint import pprint

# Where we are getting our json data from
BEER_DATA_DIRECTORY = os.path.abspath("beer_data/")
# Where are we outputting our dataset file
FLATTENED_BEER_DATA_DIRECTORY = os.path.abspath("flattened_beer_data/")

# What is the target we will be labeling, and eventually classifying, our data by?
# This should be an attribute in the flattened json of a beer object
TARGET_VALUE = 'style_name'


def generate_beer_dataset():
    """
    Generates a flattened_beer_data.json file that contains a dataset dictionary object.
    The dataset dict consists of two lists, data and labels.
    Data represents a list of flattened beer json objects.
    Labels are a list of corresponding labels for each object.
    Indexes correspond between the two lists, so labels[5] is data[5]'s label for supervised learning algorithms.
    :return:
    """

    if not os.path.exists(FLATTENED_BEER_DATA_DIRECTORY):
        os.makedirs(FLATTENED_BEER_DATA_DIRECTORY)

    dataset = {'data': [], 'labels': []}

    # Iterate over all beer data files in given directory
    outfile_name = os.path.join(FLATTENED_BEER_DATA_DIRECTORY, "flattened_beer_data.json")
    with open(outfile_name, 'w') as outfile:

        for file in os.listdir(BEER_DATA_DIRECTORY):
            with open(os.path.join(BEER_DATA_DIRECTORY, file), 'r') as infile:
                json_data = json.load(infile)
                for beer in json_data['data']:
                    # Flatten beer json, then assign data to it's target value
                    flattened_beer = flatten_json_data(beer)
                    if flattened_beer is not None: # else ignore if it doesn't have a style
                        dataset['data'].append(flattened_beer), dataset['labels'].append(flattened_beer[TARGET_VALUE])

        if len(dataset['data']) != len(dataset['labels']):
            print("ERROR: Number of data objects does not match number of labels!")

        print("Flattened output with style labels pushed to {}".format(outfile_name))
        print("Number of flattened beer objects: {}".format(len(dataset['data'])))

        outfile.write(json.dumps(dataset))


def flatten_json_data(beer_json):
    """
    Takes a single beer json object in the BreweryDB API format, and flattens and condenses to be
    added into a dataset.
    :param beer_json:
    :return:
    """
    if 'available' in beer_json:
        if 'name' in beer_json['available']:
            beer_json['available_name'] = beer_json['available']['name']
        del beer_json['available']
    if 'glass' in beer_json:
        if 'name' in beer_json['glass']:
            beer_json['glass_name'] = beer_json['glass']['name']
        del beer_json['glass']
    if 'style' in beer_json:
        if 'abvMax' in beer_json['style']:
            beer_json['style_abvMax'] = beer_json['style']['abvMax']
        if 'abvMin' in beer_json['style']:
            beer_json['style_abvMin'] = beer_json['style']['abvMin']
        if 'category' in beer_json['style']:
            beer_json['style_category_name'] = beer_json['style']['category']['name']
        if 'description' in beer_json['style']:
            beer_json['style_description'] = beer_json['style']['description']
        if 'fgMax' in beer_json['style']:
            beer_json['style_fgMax'] = beer_json['style']['fgMax']
        if 'fgMin' in beer_json['style']:
            beer_json['style_fgMin'] = beer_json['style']['fgMin']
        if 'ibuMax' in beer_json['style']:
            beer_json['style_ibuMax'] = beer_json['style']['ibuMax']
        if 'ibuMin' in beer_json['style']:
            beer_json['style_ibuMin'] = beer_json['style']['ibuMin']
        if 'name' in beer_json['style']:
            beer_json['style_name'] = beer_json['style']['name']
        if 'ogMin' in beer_json['style']:
            beer_json['style_ogMin'] = beer_json['style']['ogMin']
        if 'ogMax' in beer_json['style']:
            beer_json['style_ogMax'] = beer_json['style']['ogMax']
        if 'shortName' in beer_json['style']:
            beer_json['style_shortName'] = beer_json['style']['shortName']
        if 'srmMax' in beer_json['style']:
            beer_json['style_srmMax'] = beer_json['style']['srmMax']
        if 'srmMin' in beer_json['style']:
            beer_json['style_srmMin'] = beer_json['style']['srmMin']
        del (beer_json['style'])
    else:
        beer_json = None

    return beer_json


def trim_data(flattened_beer_json, trim_level):
    """
    Given a trim_level, trim the number of beers in a flattened beer json file to the designated
    attributes of interest chosen by the trim level. Only beers that have all of the attributes in the
    selected set will be kept, and all other attributes in those beers will be deleted (therefore
    we are filtering out bad data and smoothing dimensionality in one sweep).

    :param flattened_beer_json: This should be a python object representation of the json!
    :param trim_level: Chooses our set of attributes we wish to filter by
    :return:
    """

    attributes_of_interest = {
        0: ['style_category_name', 'style_ibuMin', 'style_ibuMax', 'style_abvMin', 'style_abvMax'],
        1: ['abv', 'ibu']
    }

    trimmed = {'data': [], 'labels': []}

    for index, beer in enumerate(flattened_beer_json['data']):
        # if all attributes of interest are present
        if all(attribute in beer for attribute in attributes_of_interest[trim_level]):
            for attribute in list(beer):
                if attribute not in attributes_of_interest[trim_level]:
                    del (beer[attribute])
            trimmed['data'].append(beer)
            trimmed['labels'].append(flattened_beer_json['labels'][index])

    return trimmed


# python data_transformer.py
if __name__ == "__main__":
    generate_beer_dataset()
