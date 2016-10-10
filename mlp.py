"""
mlp.py

Multi-layer perceptron (deep neural network) test file.
"""
from sklearn import datasets, neural_network
import matplotlib.pyplot as plot    # for viewing the numbers

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

print(X_digits)
print(y_digits)

n_samples = len(X_digits)

X_train = X_digits[:.5 * n_samples]
y_train = y_digits[:.5 * n_samples]
X_test = X_digits[.5 * n_samples:]
y_test = y_digits[.5 * n_samples:]

# hidden_layer_sizes=(100, 100, 100) ==> Neural network with 3 hidden layers of 100 units each (5 total layers)
# max_iter is the number of iterations we will stop the simulations if it doesn't converge by that point
mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

print('MLP score: {}'.format(mlp_classifier.fit(X_train, y_train).score(X_test, y_test)))
print("Predicted probabilities: \n{}".format(mlp_classifier.predict_proba(digits.data[-1])))
print("Predicted outcome: {}".format(mlp_classifier.predict(digits.data[-1])))
plot.figure(1, figsize=(3, 3))
plot.imshow(digits.images[-1], cmap=plot.cm.gray_r, interpolation='nearest')
plot.show()