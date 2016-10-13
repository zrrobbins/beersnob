"""
main.py

Main script for running classifiers on beer data.

Usage:
python main.py [trim_level]

"""
import os
import sys
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import warnings

from data_transformer import trim_data
from data_transformer import FLATTENED_BEER_DATA_DIRECTORY

# 1 = no cross validation, split based on TEST_DATA
# >1 = cross validate with CROSS_VALIDATE number of folds
CROSS_VALIDATE = 1
# Percent test data
TEST_DATA = .3
# Trim attribute set, refer to trim_data in data_transformer.py for description of attributes in each set
TRIM_SELECTION = 4

# If trim specified on command line, override value above (useful for easy testing)
if len(sys.argv) > 1 and len(sys.argv) < 3:
    TRIM_SELECTION = int(sys.argv[1])


names = ["Nearest Neighbors w/ 5 Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest (10 Trees)", "Random Forest (100 Trees)",
         "Neural Net (100,)", "Neural Net (100, 100)",
         "Neural Net (100, 100, 100)", "AdaBoost", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000),
    MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()
]


with open(os.path.join(FLATTENED_BEER_DATA_DIRECTORY, 'flattened_beer_data.json'), 'r') as infile:
    json_data = infile.read()
    json_data = json.loads(json_data)
    # Consider only passing slices of the data in here, or the classifier could take a VERY long time to run!
    # Some datasets take longer to classify than others (e.g. trim_level=1 takes longer than trim_level=0)
    trimmed_beers = trim_data({'data': json_data['data'], 'labels': json_data['labels']}, TRIM_SELECTION)

vectorizer = DictVectorizer(sparse=False) # Used to transform beer dict into numpy arrays so classifiers can use

beers = vectorizer.fit_transform(trimmed_beers['data'])
labels = trimmed_beers['labels']

n_samples = len(trimmed_beers['data'])
print("Trimmed data len: {}".format(n_samples))

if CROSS_VALIDATE > 1:
    print("Cross validation w/ {} folds.".format(CROSS_VALIDATE))
    warnings.filterwarnings("ignore", category=Warning) # Suppress cross validation warnings because they annoy me
else:
    print("Train/test split = {}/{}".format(1-TEST_DATA, TEST_DATA))
print("Calculating...\n")

for name, clf in zip(names, classifiers):
    if CROSS_VALIDATE > 1:
        scores = cross_val_score(clf, beers, labels, cv=CROSS_VALIDATE)
        print("{0} score: {1:.4f} (+/- {2:.4f})".format(name, scores.mean(), scores.std() * 2))
    else:
        X_train, X_test, y_train, y_test = train_test_split(beers, labels, test_size=TEST_DATA, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('{0} score: {1:.4f}'.format(name, score))
