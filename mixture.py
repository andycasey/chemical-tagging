# coding: utf-8

""" Chemical tagging using Gaussian mixture models """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import itertools

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn import mixture
from sklearn.cross_validation import StratifiedKFold

# XDGMM
import astroML.density_estimation
from astroML.plotting.tools import draw_ellipse

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


def xdgmm():

	all_data = oc_data

	# Get the data in a format for XDGMM

	elements = ("[Fe/H]", "[Mg/Fe]")

	data = np.zeros(map(len, (all_data, elements)))
	uncertainties = np.zeros(map(len, (all_data, elements, elements)))

	# Fill in the data
	for i, element in enumerate(elements):
		data[:, i] = all_data[element]

	# Fill in the uncertainties
	diag = np.arange(len(elements))
	for i, row in enumerate(all_data):
		uncertainties[i][diag, diag] = \
			[row["e_{0}".format(element)] for element in elements]

	# Where the data are missing (e.g. non-finite), set the value
	# as zero and the uncertainty as very large.
	data[~np.isfinite(data)] = 0
	uncertainties[~np.isfinite(uncertainties)] = 1000

	classifier = astroML.density_estimation.XDGMM(n_components=20, n_iter=500)
	classifier.fit(data, uncertainties)

	# Sample the classifier
	samples = classifier.sample(len(all_data))

	# Plot some results
	fig = plt.figure(figsize=(4.025, 7.70))
	fig.subplots_adjust(left=0.20, bottom=0.07, right=0.95, top=0.95,
		wspace=0.20, hspace=0.05)

	# Plot observed data
	ax_observed = fig.add_subplot(311)
	ax_observed.errorbar(all_data[elements[0]], all_data[elements[1]],
		fmt=None, ecolor="#666666", xerr=all_data["e_" + elements[0]],
		yerr=all_data["e_" + elements[1]], zorder=-1)
	ax_observed.scatter(all_data[elements[0]], all_data[elements[1]],
		facecolor="k")

	ax_observed.text(0.05, 0.95, "Observed Data", ha="left", va="top",
		transform=ax_observed.transAxes)

	ax_observed.set_xlim(-0.5, 0.5)
	ax_observed.set_ylim(-0.5, 0.5)
	ax_observed.set_xticklabels([""] * len(ax_observed.get_xticks()))
	ax_observed.yaxis.set_major_locator(MaxNLocator(5))

	ax_observed.set_ylabel(elements[1])

	# Plot sampled data
	ax_sampled = fig.add_subplot(312, sharex=ax_observed, sharey=ax_observed)
	ax_sampled.scatter(samples[:, 0], samples[:, 1],
		facecolor="k")
	ax_sampled.text(0.05, 0.95, "Extreme Deconvolution\nresampling",
		ha="left", va="top", transform=ax_sampled.transAxes)

	ax_sampled.set_xticklabels([""] * len(ax_sampled.get_xticks()))
	ax_sampled.set_ylabel(elements[1])

	# Plot cluster data
	ax_clusters = fig.add_subplot(313, sharey=ax_observed)
	ax_clusters.text(0.05, 0.95, "Clusters", ha="left", va="top",
		transform=ax_clusters.transAxes)

	for i in range(classifier.n_components):
		draw_ellipse(classifier.mu[i], classifier.V[i], scales=[2],
			ax=ax_clusters, ec='k', fc='gray', alpha=0.2)

	ax_clusters.set_xlim(ax_sampled.get_xlim())
	ax_clusters.set_xlabel(elements[0])
	ax_clusters.set_ylabel(elements[1])

	raise a


if __name__ == "__main__":
	#	example()
	None


