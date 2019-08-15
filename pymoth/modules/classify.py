#!/usr/bin/env python3

"""

.. module:: classify
   :platform: Unix
   :synopsis: Classify output from MothNet model.

.. moduleauthor:: Adam P. Jones <ajones173@gmail.com>

"""

from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from scipy import interp

def roc_multi(true_classes, likelihoods):
	"""

	Measure ROC AUC for multi-class classifiers.

	Params:
		true_classes (numpy array): class labels [observations,]
		likelihoods (numpy array): predicted likelihoods [observations x classes]

	Returns:
		output (dict):
			- targets (numpy array): one-hot-encoded target labels
			- roc_auc (dict): ROC curve and ROC area for each class
			- fpr (dict): false-positive rate for each class
			- tpr (dict): true-positive rate for each class

	>>> roc_dict = roc_multi(true_classes, likelihoods)

	"""

	n_classes = len(set(true_classes))

	# one-hot-encode target labels
	targets = np.eye(n_classes)[true_classes.astype(int)]

	# compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in set(true_classes.astype(int)):
		fpr[i], tpr[i], _ = roc_curve(targets[:,i], likelihoods[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(), likelihoods.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	## compute macro-average ROC curve and ROC area
	# first aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# finally, average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	output = dict()
	output['targets'] = targets
	output['roc_auc'] = roc_auc
	output['fpr'] = fpr
	output['tpr'] = tpr
	return output

def classify_digits_log_likelihood(results):
	"""
	Classify the test digits in a run using log likelihoods from the various EN responses.

	Steps:
		#. for each test digit (ignore non-postTrain digits), for each EN, calculate \
		the number of stds the test digit is from each class distribution. This makes \
		a 10 x 10 matrix where each row corresponds to an EN, and each column corresponds \
		to a class.
		#. Square this matrix by entry. Sum the columns. Select the col with the lowest \
		value as the predicted class. Return the vector of sums in 'likelihoods'.
		#. The rest is simple calculation.

	Args:
		results (dict): output from :func:`simulate`. i'th entry gives results for all \
		classes, in the _i_th EN.

	Returns:
		output (dict):
			- true_classes (numpy array): shortened version of whichOdor (with only \
			post-training, ie validation, entries)
			- targets (numpy array): one-hot-encoded target labels
			- roc_auc (dict): ROC curve and ROC area for each class
			- fpr (dict): false-positive rate for each class
			- tpr (dict): true-positive rate for each class
			- pred_classes (numpy array): predicted classes
			- likelihoods (numpy array): [n x 10] each row a post_training digit \
			(entries are summed log likelihoods)
			- acc_perc (numpy array): [n x 10] class accuracies as percentages
			- total_acc (float): overall accuracy as percentage
			- conf_mat (numpy array): i,j'th entry is number of test digits with true \
			label i that were predicted to be j

	>>> classify_digits_log_likelihood( dummy_results )
	"""

	n_en = len(results) # number of ENs, same as number of classes
	pre_train_inds = np.nonzero(results[1]['post_train_resp'] >= 0)[0] # indices of post-train (ie validation) digits
	# TO DO: Why use 2 (1, here) as index above? Ask CBD
	n_post = len(pre_train_inds) # number of post-train digits

	# extract true classes (digits may be referred to as odors or 'odor puffs'):
	true_classes = results[0]['odor_class'][pre_train_inds]
	# TO DO: Why use 1 (0, here) as index above? Ask CBD

	# extract the relevant odor puffs: Each row is an EN, each col is an odor puff
	post_train_resp = np.full((n_en,n_post), np.nan)
	for i,resp in enumerate(results):
		post_train_resp[i,:] = resp['post_train_resp'][pre_train_inds]

	# make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class:
	mu = np.full((n_en,n_en), np.nan)
	sig = np.full((n_en,n_en), np.nan)
	for i,resp in enumerate(results):
		mu[i,:] = resp['post_mean_resp']
		sig[i,:] = resp['post_std_resp']

	# for each EN:
	# get the likelihood of each puff (ie each col of post_train_resp)
	likelihoods = np.zeros((n_post,n_en))
	for i in range(n_post):
		# Caution: post_train_resp[:,i] becomes a row vector, but we need it to stay as a
		# col vector so we can make 10 identical columns. So transpose it back with [np.newaxis]
		a = post_train_resp[:,i][np.newaxis]
		dist = ( np.tile( a.T, ( 1, 10 )) - mu) / sig # 10 x 10 matrix
		# The ith row, jth col entry is the mahalanobis distance of this test
		# digit's response from the i'th ENs response to the j'th class.
		# For example, the diagonal contains the mahalanobis distance of this
		# digit's response to each EN's home-class response.

		likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener

	# make predictions:
	pred_classes = np.argmin(likelihoods, axis=1)

	# calc accuracy percentages:
	class_acc = np.zeros(n_en)
	for i in range(n_en):
		class_acc[i] = (100*np.logical_and(pred_classes == i, true_classes == i).sum())/(true_classes == i).sum()

	total_acc = (100*(pred_classes == true_classes).sum())/len(true_classes)

	# calc confusion matrix:
	# i,j'th entry is number of test digits with true label i that were predicted to be j
	confusion = confusion_matrix(true_classes, pred_classes)

	# measure ROC AUC for each class
	roc_dict = roc_multi(true_classes, likelihoods*-1)

	return {
		'true_classes':true_classes,
		'targets':roc_dict['targets'],
		'roc_auc':roc_dict['roc_auc'],
		'fpr':roc_dict['fpr'],
		'tpr':roc_dict['tpr'],
		'pred_classes':pred_classes,
		'likelihoods':likelihoods,
		'acc_perc':class_acc,
		'total_acc':total_acc,
		'conf_mat':confusion,
			}


def classify_digits_thresholding(results, home_advantage, home_thresh_sigmas, above_home_thresh_reward):
	"""
	Classify the test digits using log likelihoods from the various EN responses, \
	with the added option of rewarding high scores relative to an ENs home-class \
	expected response distribution.
	One use of this function is to apply de-facto thresholding on discrete ENs, \
	so that the predicted class corresponds to the EN that spiked most strongly \
	(relative to its usual home-class response).

	Steps:
		#. For each test digit (ignore non-postTrain digits), for each EN, calculate \
		the # stds from the test digit is from each class distribution. This makes \
		a 10 x 10 matrix where each row corresponds to an EN, and each column \
		corresponds to a class.
		#. Square this matrix by entry. Sum the columns. Select the col with the \
		lowest value as the predicted class. Return the vector of sums in 'likelihoods'.
		#. The rest is simple calculation.

	Args:
		results (dict): [1 x 10] dict produced by :func:`collect_stats`.
		home_advantage (int): the emphasis given to the home EN. It multiplies the \
		off-diagonal of dist. 1 -> no advantage (default). Very high means that a \
		test digit will be classified according to the home EN it does best in, \
		ie each EN acts on its own.
		home_thresh_sigmas (int): the number of stds below an EN's home-class mean \
		that we set a threshold, such that if a digit scores above this threshold \
		in an EN, that EN will be rewarded by 'above_home_thresh_reward'.
		above_home_thresh_reward (int): if a digit's response scores above the EN's \
		mean home-class value, reward it by dividing by this value. This reduces \
		the log likelihood score for that EN.

	Returns:
		output (dict):
			- true_classes (numpy array): shortened version of whichOdor (with only \
			- post-training, ie validation, entries)
			- targets (numpy array): one-hot-encoded target labels
			- roc_auc (dict): ROC curve and ROC area for each class
			- fpr (dict): false-positive rate for each class
			- tpr (dict): true-positive rate for each class
			- pred_classes (numpy array): predicted classes
			- likelihoods (numpy array): [n x 10] each row a post_training digit \
			(entries are summed log likelihoods)
			- acc_perc (numpy array): [n x 10] class accuracies as percentages
			- total_acc (float): overall accuracy as percentage
			- conf_mat (numpy array): i,j'th entry is number of test digits with true \
			label i that were predicted to be j
			- home_advantage (int): the emphasis given to the home EN. It multiplies the \
			off-diagonal of dist. 1 -> no advantage (default). Very high means that a \
			test digit will be classified according to the home EN it does best in, \
			ie each EN acts on its own.
			- home_thresh_sigmas (int): the number of stds below an EN's home-class mean \
			that we set a threshold, such that if a digit scores above this threshold \
			in an EN, that EN will be rewarded by 'above_home_thresh_reward'.

	>>> classify_digits_thresholding( dummy_results )

	"""

	n_en = len(results) # number of ENs, same as number of classes
	pre_train_inds = np.nonzero(results[1]['post_train_resp'] >= 0)[0] # indices of post-train (ie validation) digits
	# DEV NOTE: Why use 2 (1, in Python) as index above? Ask CBD
	n_post = len(pre_train_inds) # number of post-train digits

	# extract true classes:
	true_classes = results[0]['odor_class'][pre_train_inds] # throughout, digits may be referred to as odors or 'odor puffs'
	# DEV NOTE: Why use 1 (0, in Python) as index above? Ask CBD

	# extract the relevant odor puffs: Each row is an EN, each col is an odor puff
	post_train_resp = np.full((n_en,n_post), np.nan)
	for i,resp in enumerate(results):
		post_train_resp[i,:] = resp['post_train_resp'][pre_train_inds]

	# make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class.
	# For example, the i'th row, j'th col entry of 'mu' is the mean of the i'th
	# EN in response to digits from the j'th class; the diagonal contains the
	# responses to the home-class.
	mu = np.full((n_en,n_en), np.nan)
	sig = np.full((n_en,n_en), np.nan)
	for i,resp in enumerate(results):
		mu[i,:] = resp['post_mean_resp']
		sig[i,:] = resp['post_std_resp']

	# for each EN:
	# get the likelihood of each puff (ie each col of post_train_resp)
	likelihoods = np.zeros((n_post,n_en))
	for i in range(n_post):

		dist = (np.tile(post_train_resp[:,i],(10,1)) - mu) / sig # 10 x 10 matrix
		# The ith row, jth col entry is the mahalanobis distance of this test
		# digit's response from the i'th ENs response to the j'th class.
		# For example, the diagonal contains the mahalanobis distance of this
		# digit's response to each EN's home-class response.

		# 1. Apply rewards for above-threshold responses:
		off_diag = dist - np.diag(np.diag(dist))
		on_diag = np.diag(dist).copy()
		# Reward any onDiags that are above some threshold (mu - n*sigma) of an EN.
		# CAUTION: This reward-by-shrinking only works when off-diagonals are
		# demolished by very high value of 'home_advantage'.
		home_threshs = home_thresh_sigmas * np.diag(sig)
		# aboveThreshInds = np.nonzero(on_diag > home_threshs)[0]
		on_diag[on_diag > home_threshs] /= above_home_thresh_reward
		on_diag = np.diag(on_diag) # turn back into a matrix
		# 2. Emphasize the home-class results by shrinking off-diagonal values.
		# This makes the off-diagonals less important in the final likelihood sum.
		# This is shrinkage for a different purpose than in the lines above.
		dist = (off_diag / home_advantage) + on_diag
		likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener
		# In pure thresholding case (ie off-diagonals ~ 0), this does not matter.

	# make predictions:
	pred_classes = np.argmin(likelihoods, axis=1)
	# for i in range(n_post):
		# pred_classes[i] = find(likelihoods(i,:) == min(likelihoods(i,:) ) )

	# calc accuracy percentages:
	class_acc = np.zeros(n_en)
	for i in range(n_en):
		class_acc[i] = (100*np.logical_and(pred_classes == i, true_classes == i).sum())/(true_classes == i).sum()

	total_acc = (100*(pred_classes == true_classes).sum())/len(true_classes)

	# confusion matrix:
	# i,j'th entry is number of test digits with true label i that were predicted to be j
	confusion = confusion_matrix(true_classes, pred_classes)

	# measure ROC AUC for each class
	roc_dict = roc_multi(true_classes, likelihoods)

	return {
		'true_classes':true_classes,
		'targets':roc_dict['targets'],
		'roc_auc':roc_dict['roc_auc'],
		'fpr':roc_dict['fpr'],
		'tpr':roc_dict['tpr'],
		'pred_classes':pred_classes,
		'likelihoods':likelihoods,
		'acc_perc':class_acc,
		'total_acc':total_acc,
		'conf_mat':confusion,
		'home_advantage':home_advantage,
		'home_thresh_sigmas':home_thresh_sigmas,
			}

# MIT license:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
