# coding: utf-8

""" Chemical tagging using Gaussian mixture models """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import itertools

# Third party imports
import numpy as np
import matplotlib.pyplot as plt


from sklearn import mixture
from sklearn.cross_validation import StratifiedKFold

# Module imports
import dataio

oc_data = dataio.get_mitschang_data()

def example():
	n_samples = 300

	# generate random sample, two components
	np.random.seed(0)
	C = np.array([[0., -0.7], [3.5, .7]])

	X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C),
	                np.random.randn(n_samples, 2) + np.array([20, 20])]

	clf = mixture.GMM(n_components=2, covariance_type='full')
	clf.fit(X_train)

	x = np.linspace(-20.0, 30.0)
	y = np.linspace(-20.0, 40.0)
	X, Y = np.meshgrid(x, y)
	XX = np.c_[X.ravel(), Y.ravel()]
	Z = np.log(-clf.score_samples(XX)[0])
	Z = Z.reshape(X.shape)

	CS = plt.contour(X, Y, Z)
	CB = plt.colorbar(CS, shrink=0.8, extend='both')
	plt.scatter(X_train[:, 0], X_train[:, 1], .8)

	plt.axis('tight')
	plt.show()


def classify(elements):

	# Data needs to be of shape (measurements, dimensions)
	chemical_data = np.zeros((len(oc_data), len(elements)))
	for i, element in enumerate(elements):
		chemical_data[:, i] = oc_data[element]

	cluster_names = list(set(oc_data["cluster"]))
	np.random.shuffle(cluster_names)

	clusters = np.array([cluster_names.index(cluster) \
		for cluster in oc_data["cluster"]])

	# Only supply finite rows
	is_finite = np.all(np.isfinite(chemical_data), axis=1)
	chemical_data = chemical_data[is_finite]
	clusters = clusters[is_finite]

	# Split the data
	skf = StratifiedKFold(chemical_data, n_folds=2)
	train_index, test_index = next(iter(skf))

	data_train = chemical_data[train_index]
	clusters_train = clusters[train_index]
	data_test = chemical_data[test_index]
	clusters_test = clusters[test_index]

	num_classes = len(np.unique(clusters_train))


	classifier = mixture.DPGMM(n_components=num_classes, covariance_type="tied")

	classifier.means_ = np.array([data_train[clusters_train == cluster].mean(axis=0) \
								  for cluster in np.unique(clusters_train)])
	classifier.fit(data_train)


	cluster_train_pred = classifier.predict(data_train)
	train_accuracy = np.mean(cluster_train_pred.ravel() == clusters_train.ravel()) * 100

	cluster_test_pred = classifier.predict(data_test)
	test_accuracy = np.mean(cluster_test_pred.ravel() == clusters_test.ravel()) * 100

	print("Train accuracy: {0:5.1f} Test accuracy: {1:5.1f} with {2} clusters, {3} stars and elements {4}".format(
		train_accuracy, test_accuracy, num_classes, len(chemical_data), ", ".join(elements)))

	return (train_accuracy, test_accuracy, len(chemical_data), num_classes)


def classify_all():

	results = []
	combinations = []
	all_elements = [name for name in oc_data.dtype.names if name.startswith("[")]

	for i in xrange(2):

		for combination in itertools.combinations(all_elements, i + 1):
			try:
				train_accuracy, test_accuracy, num_points, num_clusters = classify(combination)

			except:
				success, train_accuracy, test_accuracy = 0, 0, 0
				num_points, num_clusters = 0, 0

			else:
				success = 1

			combinations.append(combination)
			results.append([train_accuracy, test_accuracy, success, num_points, num_clusters])

	results = np.core.records.fromrecords(results,
		names=",".join(["train", "test", "success", "points", "clusters"]),
		formats=",".join(["f8"] * 5))

	return results



if __name__ == "__main__":
	#	example()
	None


