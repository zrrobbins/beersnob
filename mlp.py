"""
mlp.py

Multi-layer perceptron (deep neural network) test file.
"""
import os
import json

from sklearn import neural_network
from sklearn.feature_extraction import DictVectorizer
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from data_transformer import trim_data
from data_transformer import FLATTENED_BEER_DATA_DIRECTORY

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

TRAINING_DATA = .5 # Percent training data

with open(os.path.join(FLATTENED_BEER_DATA_DIRECTORY, 'flattened_beer_data.json'), 'r') as infile:
    json_data = infile.read()
    json_data = json.loads(json_data)
    # Consider only passing slices of the data in here, or the classifier could take a VERY long time to run!
    # Some datasets take longer to classify than others (e.g. trim_level=1 takes longer than trim_level=0)
    trimmed_beers = trim_data({'data': json_data['data'], 'labels': json_data['labels']}, 3)

vectorizer = DictVectorizer(sparse=False) # Used to transform beer dict into numpy arrays so classifiers can use

beers = vectorizer.fit_transform(trimmed_beers['data'])
labels = trimmed_beers['labels']

n_samples = len(trimmed_beers['data'])
print("Trimmed data len: {}".format(n_samples))

X_train = beers[:int(TRAINING_DATA * n_samples)]
y_train = labels[:int(TRAINING_DATA * n_samples)]
X_test = beers[int(TRAINING_DATA * n_samples):]
y_test = labels[int(TRAINING_DATA * n_samples):]

# TODO: Try other classifiers (i.e. Naive Bayes, KNN, SVM, Decision Trees, etc.)

print("Calculating...\n")
# hidden_layer_sizes=(100, 100, 100) ==> Neural network with 3 hidden layers of 100 units each (5 total layers)
# max_iter is the number of iterations we will stop the simulations if it doesn't converge by that point
mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)

print('MLP score: {}'.format(mlp_classifier.fit(X_train, y_train).score(X_test, y_test)))
print("Predicted probabilities for beer types: \n{}".format(mlp_classifier.predict_proba(beers[-1].reshape(1,-1))))
print("Predicted outcome for last beer: {}".format(mlp_classifier.predict(beers[-1].reshape(1,-1))))
print("Actual outcome for last beer: {}".format(labels[-1]))
