'''
runMothLearnerMNIST

Main script to train a moth brain model on a crude (downsampled) MNIST set.
The moth can be generated from template or loaded complete from file.

Modifying parameters:
	1. Modify 'ModelParams' with the desired parameters for generating a moth
	(ie neural network), or specify a pre-existing 'model_params' file to load.
	2. Edit USER ENTRIES

The dataset:
	Because the moth brain architecture, as evolved, only handles ~60 features,
we need to create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
We do this by cropping and downsampling the mnist thumbnails, then selecting a subset
of the remaining pixels.
	This results in a cruder dataset (set various view flags to see thumbnails).
However, it is sufficient for testing the moth brain's learning ability. Other ML
methods need to be tested on this same cruder dataset to make useful comparisons.

Define train and control pools for the experiment, and determine the receptive field.
This is done first because the receptive field determines the number of AL units, which
     must be updated in model_params before 'init_connection_matrix' runs.
This dataset will be used for each simulation in numRuns. Each simulation draws
	a new set of samples from this set.

Order of events:
	1. Load and pre-process dataset
	Within the loop over number of simulations:
	2. Select a subset of the dataset for this simulation (only a few samples are used)
	3. Create a moth (neural net). Either select an existing moth file, or generate a new moth in 2 steps:
		a) run 'ModelParams' and incorporate user entry edits such as 'goal'
		b) create connection matrices via 'init_connection_matrix'
	4. Load the experiment parameters
	5. Run the simulation with 'sde_wrap', print results to console
	6. Plot results (optional)
	7. Run addition ML models for comparison, print results to console (optional)

Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
MIT License
'''

# import packages
import time
runStart = time.time() # time execution duration
import numpy as np
import os
import copy # for deep copy of nested lists
import sys

##TEST for Python version > 2
python_version = "{}.{}".format(sys.version_info.major,sys.version_info.minor)
if sys.version_info.major > 2:
	print("Python version {} detected.".format(python_version))
else:
	version_error = "Python version {} detected.\n".format(python_version) + \
					"Python version 3 or higher is required to run this module.\n" + \
					"Please install Python 3+."
	raise Exception(version_error)

# Experiment details
from support_functions.generate import generate_ds_MNIST
from support_functions.show_figs import show_FA_thumbs, show_EN_resp, show_roc_curves
from support_functions.params import ExpParams
from support_functions.sde import sde_wrap
from support_functions.classify import classify_digits_log_likelihood, classify_digits_thresholding
from support_functions.params import ModelParams

### 1. Object initialization ###

## USER ENTRIES (Edit parameters below):
#-------------------------------------------------------------------------------
screen_size = (1920, 1080) # screen size (width, height)

# use_existing_conn_matrices = False
## if True, load 'matrixParamsFilename', which includes filled-in connection matrices
## if False, generate new moth from template in params.py

# matrix_params_filename = 'sampleMothModelParams'
## dict with all info, including connection matrices, of a particular moth

num_runs = 1 # how many runs you wish to do with this moth or moth template,
# each run using random draws from the mnist set

goal  = 15
# defines the moth's learning rates, in terms of how many training samples per
# class give max accuracy. So "goal = 1" gives a very fast learner.
# if goal == 0, the rate parameters defined the template will be used as-is
# if goal > 1, the rate parameters will be updated, even in a pre-set moth

tr_per_class = 3 # (try 3) the number of training samples per class
num_sniffs = 2 # (try 2) number of exposures each training sample

# nearest neighbors
run_nearest_neighbors = True # this option requires the sklearn library be installed
num_neighbors = 1 # hyper param for nearest neighbors
# Suggested values: tr_per_class ->
#	num_neighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

# SVM
runSVM = True # this option requires the sklearn library be installed
box_constraint = 1e1 # optimization parameter for svm
# Suggested values: tr_per_class ->
#	box_constraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1,
#					20 -> 1e-4 or 1e-5, 50 -> 1e-5 ; 100+ -> 1e-7

## Flags to show various images:
n_thumbnails = 1 # N means show N experiment inputs from each class
	# 0 means don't show any

# To save results if wished:
save_all_neural_timecourses = False # 0 -> save only EN (ie readout) timecourses
# Caution: 1 -> very high memory demands, hinders longer runs

# flag for statistical plots of EN response changes: One image (with 8 subplots) per EN
show_acc_plots = True # True to plot, False to ignore
# flag for EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)
show_time_plots = True # True to plot, False to ignore
save_results_folder = 'results' # String (relative path)
# If non-empty, results will be saved here

results_filename = 'results' # will get the run number appended to it

#-------------------------------------------------------------------------------

# Test parameters for compatibility
if run_nearest_neighbors or runSVM:
	##TEST to see if sklearn is installed,
	try:
	    import sklearn
	except ImportError:
	    print('sklearn is not installed, and it is required to run ML models.\n' + \
			"Install it or set run_nearest_neighbors and runSVM to 'False'.")

if show_acc_plots or show_time_plots:
	##TEST that directory string is not empty
	if not save_results_folder:
		folder_error = "save_results_folder parameter is empty.\n" + \
			"Please add directory or set show_acc_plots and show_time_plots to 'False'."
		raise Exception(folder_error)

	##TEST for existence of image results folder, else create it
	if not os.path.isdir(save_results_folder):
		os.mkdir('./'+save_results_folder)
		print('Creating results directory: {}'.format(os.path.join(os.getcwd(),save_results_folder)))

#-------------------------------------------------------------------------------

### 2. Load and preprocess MNIST dataset ###
# The dataset:
# Because the moth brain architecture, as evolved, only handles ~60 features, we need to
# create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
# We do this by cropping and downsampling the MNIST thumbnails, then selecting
# a subset of the remaining pixels.
# This results in a cruder dataset (set various view flags to see thumbnails).
# However, it is sufficient for testing the moth brain's learning ability. Other
# ML methods need to be tested on this same cruder dataset to make useful comparisons.

# Define train and control pools for the experiment, and determine the receptive field.
# This is done first because the receptive field determines the number of AL units, which
#      must be updated in model_params before 'initializeMatrixParams_fn' runs.
# This dataset will be used for each simulation in num_runs. Each
#      simulation draws a new set of samples from this set.

# Parameters required for the dataset generation function:
# 1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
#   This is NOT the number of training samples per class.
# 	That is tr_per_class, defined above.

class_labels = np.array(range(10))  # For MNIST. '0' is labeled as 10
val_per_class = 15  # number of digits used in validation sets and in baseline sets

# make a vector of the classes of the training samples, randomly mixed:
tr_classes = np.repeat( class_labels, tr_per_class )
tr_classes = np.random.permutation( tr_classes )
# repeat these inputs if taking multiple sniffs of each training sample:
tr_classes = np.tile( tr_classes, [1, num_sniffs] )[0]

# Specify pools of indices from which to draw baseline, train, val sets.
ind_pool_baseline = list(range(100)) # 1:100
ind_pool_train = list(range(100,300)) # 101:300
ind_pool_post = list(range(300,400)) # 301:400

## Create preprocessing parameters
# Population pre-processing pools of indices:
inds_to_ave = list(range(550,1000))
inds_to_calc_RF = list(range(550,1000))
max_ind = max( [ inds_to_calc_RF + ind_pool_train ][0] ) # we'll throw out unused samples

## 2. Pre-processing parameters for the thumbnails:
downsample_rate = 2
crop = 2
num_features = 85 # number of pixels in the receptive field
pixel_sum = 6
show_thumbnails = n_thumbnails
downsample_method = 1 # 0 means sum square patches of pixels
					# 1 means use bicubic interpolation

# generate the data array:
# The dataset fA is a feature array ready for running experiments.
# Each experiment uses a random draw from this dataset.
fA, active_pixel_inds, len_side = generate_ds_MNIST('',
	max_ind, class_labels, crop, downsample_rate, downsample_method, inds_to_ave,
	pixel_sum, inds_to_calc_RF, num_features, screen_size, save_results_folder,
	show_thumbnails
	)

_, num_per_class, class_num = fA.shape
# fA = n x m x 10 array where n = #active pixels, m = #digits from each class
# that will be used. The 3rd dimension gives the class: 0:9.

def setup_digit_queues(fA):
	''' Subsample the dataset for this simulation '''
	# Line up the images for the experiment (in 10 parallel queues)
	digit_queues = np.zeros_like(fA)

	for i in class_labels:

		## 1. Baseline (pre-train) images
		# choose some images from the baselineIndPool
		range_top_end = max(ind_pool_baseline) - min(ind_pool_baseline) + 1
		r_sample = np.random.choice(range_top_end, val_per_class) # select random digits
		these_inds = min(ind_pool_baseline) + r_sample
		digit_queues[:,:val_per_class,i] = fA[:,these_inds,i]

		## 2. Training images
		# choose some images from the trainingIndPool
		range_top_end = max(ind_pool_train) - min(ind_pool_train) + 1
		r_sample = np.random.choice(range_top_end, tr_per_class) # select random digits
		these_inds = min(ind_pool_train) + r_sample
		# repeat these inputs if taking multiple sniffs of each training sample
		these_inds = np.tile(these_inds, num_sniffs)
		digit_queues[:, val_per_class:(val_per_class+tr_per_class*num_sniffs), i] = fA[:, these_inds, i]

		## 3. Post-training (val) images
		# choose some images from the postTrainIndPool
		range_top_end = max(ind_pool_post) - min(ind_pool_post) + 1
		r_sample = np.random.choice(range_top_end, val_per_class) # select random digits
		these_inds = min(ind_pool_post) + r_sample
		digit_queues[:,(val_per_class+tr_per_class*num_sniffs):(val_per_class+tr_per_class*num_sniffs+val_per_class),
			i] = fA[:, these_inds, i]

	return digit_queues

def train_test_split(digit_queues):
	''' Build train and val feature matrices and class label vectors. '''
	# X = n x numberPixels;  Y = n x 1, where n = 10*tr_per_class.
	train_X = np.zeros((10*tr_per_class, fA.shape[0]))
	val_X = np.zeros((10*val_per_class, fA.shape[0]))
	train_y = np.zeros((10*tr_per_class, 1))
	val_y = np.zeros((10*val_per_class, 1))

	# populate the labels one class at a time
	for i in class_labels:
		# skip the first 'val_per_class' digits,
		# as these are used as baseline digits in the moth (formality)
		temp = digit_queues[:,val_per_class:val_per_class+tr_per_class,i]
		train_X[i*tr_per_class:(i+1)*tr_per_class,:] = temp.T
		temp = digit_queues[:,val_per_class+tr_per_class:2*val_per_class+tr_per_class,i]
		val_X[i*val_per_class:(i+1)*val_per_class,:] = temp.T
		train_y[i*tr_per_class:(i+1)*tr_per_class] = i
		val_y[i*val_per_class:(i+1)*val_per_class,:] = i

	return train_X, val_X, train_y, val_y

#-------------------------------------------------------------------------------

### 3. Run MothNet simulation ###

for run in range(num_runs):

	# Loop through the number of simulations specified:
	sim_loop_disp = 'starting sim for goal = {}, tr_per_class = {}, numSniffsPerSample = {}'
	print(sim_loop_disp.format(goal, tr_per_class, num_sniffs))

	digit_queues = setup_digit_queues(fA)

	# show the final versions of thumbnails to be used, if wished
	if n_thumbnails:
		temp_array = np.zeros((len_side, num_per_class, class_num))
		temp_array[active_pixel_inds,:,:] = digit_queues
		normalize = 1
		show_FA_thumbs(temp_array, n_thumbnails, normalize, 'Input thumbnails',
			screen_size, os.path.join(save_results_folder,'thumbnails'))

	# Train/test split: Re-organize train and val sets for classifiers
	train_X, val_X, train_y, val_y = train_test_split(digit_queues)

	## Create a new moth:
	# instantiate template params
	model_params = ModelParams( len(active_pixel_inds), goal )
	model_params.trueClassLabels = class_labels # misc parameter tagging along

	# Populate the moth's connection matrices using the model_params
	model_params.init_connection_matrix()

	# # Define the experiment parameters, including book-keeping for time-stepped
	# # 	evolutions, eg when octopamine occurs, time regions to poll for digit
	# # 	responses, windowing of firing rates, etc
	# Load experiment params, including book-keeping for time-stepped
	# 	evolutions, eg when octopamine occurs, time regions to poll for digit
	# 	responses, windowing of Firing rates, etc
	experiment_params = ExpParams( tr_classes, class_labels, val_per_class )

	#-------------------------------------------------------------------------------

	# 3. run this experiment as sde time-step evolution:
	sim_results = sde_wrap( model_params, experiment_params, digit_queues )

	#-------------------------------------------------------------------------------

	# Process the sim results to group EN responses by class and time:
	resp_orig = show_EN_resp( sim_results, model_params, experiment_params,
		show_acc_plots, show_time_plots, class_labels, screen_size,
		images_filename=os.path.join(save_results_folder, results_filename) )

	# Calculate the classification accuracy:
	# for baseline accuracy function argin, substitute pre- for post-values in resp_orig:
	resp_naive = copy.deepcopy(resp_orig)
	for i, resp in enumerate(resp_orig):
		resp_naive[i]['postMeanResp'] = resp['preMeanResp'].copy()
		resp_naive[i]['postStdResp'] = resp['preStdResp'].copy()
		resp_naive[i]['postTrainOdorResp'] = resp['preTrainOdorResp'].copy()

	# 1. Using Log-likelihoods over all ENs:
	# Baseline accuracy:
	output_naive_log_loss = classify_digits_log_likelihood( resp_naive )
	# Post-training accuracy using log-likelihood over all ENs:
	output_trained_log_loss = classify_digits_log_likelihood( resp_orig )

	print('LogLikelihood:')
	print(' Baseline (Naive) Accuracy: {}%,'.format(round(output_naive_log_loss['total_acc'])) + \
		'by class: {}%'.format(np.round(output_naive_log_loss['acc_perc'])))
	print(' Trained Accuracy: {}%,'.format(round(output_trained_log_loss['total_acc'])) + \
		'by class: {}%'.format(np.round(output_trained_log_loss['acc_perc'])))

	# 2. Using single EN thresholding:
	output_naive_thresholding = classify_digits_thresholding( resp_naive, 1e9, -1, 10 )
	output_trained_thresholding = classify_digits_thresholding( resp_orig, 1e9, -1, 10 )

	# Compute macro-average ROC curve
	show_roc_curves(output_trained_log_loss['fpr'], output_trained_log_loss['tpr'],
		output_trained_log_loss['roc_auc'], class_labels, images_filename='./results/ROC_moth')

	### 4. Run simulation with alternative models ###
	#-------------------------------------------------------------------------------

	# nearest neighbors
	if run_nearest_neighbors:

		from sklearn.neighbors import KNeighborsClassifier
		neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
		neigh.fit(train_X, train_y.ravel())
		y_hat = neigh.predict(val_X)

		# measure overall accuracy
		nn_acc = neigh.score(val_X, val_y)

		# measure accuracy for each class
		class_acc = np.zeros_like(class_labels) # preallocate
		for i in class_labels:
			inds = np.where(val_y==i)[0]
			class_acc[i] = np.round(100*np.sum( y_hat[inds]==val_y[inds].squeeze()) /
				len(val_y[inds]) )

		print('Nearest neighbor (k[# of neighbors]={}):\n'.format(num_neighbors),
	        'Trained Accuracy = {}%,'.format(np.round(100*nn_acc)),
	        'by class: {}% '.format(class_acc) )

		# # Compute macro-average ROC curve
		# show_roc_curves(fpr, tpr, roc_auc, class_labels, images_filename='./results/ROC_knn')

	#-------------------------------------------------------------------------------

	# support vector machine
	if runSVM:

		from sklearn import svm
		svm_clf = svm.SVC(gamma='scale', C=box_constraint)
		svm_clf.fit(train_X, train_y.ravel())
		y_hat = svm_clf.predict(val_X)

		# measure overall accuracy
		svm_acc = svm_clf.score(val_X, val_y)

		# measure accuracy for each class
		class_acc = np.zeros_like(class_labels) # preallocate
		for i in class_labels:
			inds = np.where(val_y==i)[0]
			class_acc[i] = np.round(100*np.sum( y_hat[inds]==val_y[inds].squeeze()) /
				len(val_y[inds]) )

		print('Support vector machine (BoxConstraint[i.e. C]={}):\n'.format(box_constraint),
	    	'Trained Accuracy = {}%,'.format(np.round(100*svm_acc)),
	        'by class: {}% '.format(class_acc))

		# # Compute macro-average ROC curve
		# show_roc_curves(fpr, tpr, roc_auc, class_labels, images_filename='./results/ROC_knn')

print('         -------------All done-------------         ')

runDuration = time.time() - runStart
print('{} executed in {:.3f} minutes'.format(__file__, runDuration/60))

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

# import pdb; pdb.set_trace()
