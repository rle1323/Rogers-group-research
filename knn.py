import numpy as np
import time
from import_images import grab_images
from generate_labels import generate_labels
from sklearn import model_selection, cluster, metrics

features = grab_images('images_data.txt')
labels = generate_labels(features)
features = features.reshape(999*12, 64*64)
labels = labels.flatten()
print('Data loaded')

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=3)
KMeans = cluster.KMeans(n_clusters=20, init='k-means++')
KMeans.fit(X_train, Y_train)
predictions = KMeans.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, predictions))