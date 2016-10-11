"""
mlp.py

Multi-layer perceptron (deep neural network) test file.
"""
import os
import json
from sklearn import neural_network
from sklearn.feature_extraction import DictVectorizer

from data_transformer import trim_heavy
from data_transformer import FLATTENED_BEER_DATA_DIRECTORY

with open(os.path.join(FLATTENED_BEER_DATA_DIRECTORY, 'flattened_beer_data.json'), 'r') as infile:
    json_data = infile.read()
    json_data = json.loads(json_data)
    # trim_heavy will return beers' ibus and abvs only with a label of the beer type
    # Consider only passing slices of the data in here, or the classifier could take a VERY long time to run!
    trimmed_beers = trim_heavy({'data': json_data['data'][:5000], 'labels': json_data['labels'][:5000]})

vectorizer = DictVectorizer(sparse=False) # Used to transform beer dict into numpy arrays so classifiers can use

beers = vectorizer.fit_transform(trimmed_beers['data'])
labels = trimmed_beers['labels']

n_samples = len(trimmed_beers['data'])
print("Trimmed data len: {}".format(n_samples))

X_train = beers[:int(.75 * n_samples)]
y_train = labels[:int(.75 * n_samples)]
X_test = beers[int(.75 * n_samples):]
y_test = labels[int(.75 * n_samples):]

print("Calculating...\n")
# hidden_layer_sizes=(100, 100, 100) ==> Neural network with 3 hidden layers of 100 units each (5 total layers)
# max_iter is the number of iterations we will stop the simulations if it doesn't converge by that point
mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

print('MLP score: {}'.format(mlp_classifier.fit(X_train, y_train).score(X_test, y_test)))
print("Predicted probabilities for beer types: \n{}".format(mlp_classifier.predict_proba(beers[-1].reshape(1,-1))))
print("Predicted outcome for last beer: {}".format(mlp_classifier.predict(beers[-1].reshape(1,-1))))
