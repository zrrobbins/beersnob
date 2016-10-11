"""
main.py

Classifier testing file.
"""
import os
import json

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

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

TRAINING_DATA = .5 # Percent training data

with open(os.path.join(FLATTENED_BEER_DATA_DIRECTORY, 'flattened_beer_data.json'), 'r') as infile:
    json_data = infile.read()
    json_data = json.loads(json_data)
    # Consider only passing slices of the data in here, or the classifier could take a VERY long time to run!
    # Some datasets take longer to classify than others (e.g. trim_level=1 takes longer than trim_level=0)
    trimmed_beers = trim_data({'data': json_data['data'][:10000], 'labels': json_data['labels'][:10000]}, 0)

vectorizer = DictVectorizer(sparse=False) # Used to transform beer dict into numpy arrays so classifiers can use

beers = vectorizer.fit_transform(trimmed_beers['data'])
labels = trimmed_beers['labels']

n_samples = len(trimmed_beers['data'])
print("Trimmed data len: {}".format(n_samples))

X_train, X_test, y_train, y_test = train_test_split(beers, labels, test_size=.25, random_state=42)

print("Calculating...\n")
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('{} score: {}'.format(name, score))

# print("Predicted probabilities for beer types: \n{}".format(mlp_classifier.predict_proba(beers[-1].reshape(1,-1))))
# print("Predicted outcome for last beer: {}".format(mlp_classifier.predict(beers[-1].reshape(1,-1))))
# print("Actual outcome for last beer: {}".format(labels[-1]))
