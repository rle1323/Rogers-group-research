import numpy as np
import time
from import_images import grab_images
from generate_labels import generate_labels
from sklearn import model_selection, linear_model, metrics

features = grab_images('images_data.txt')
labels = generate_labels(features)
print('Data loaded')

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=0)
X_train, X_test = X_train.reshape(999*24, 64*64), X_test.reshape(6*999, 64*64)
Y_train, Y_test = Y_train.flatten(), Y_test.flatten()

Perceptron = linear_model.Perceptron(penalty='l2', max_iter=1000)
Perceptron.fit(X_train, Y_train)
predictions = Perceptron.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, predictions))