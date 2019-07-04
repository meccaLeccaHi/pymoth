'''
runMothLearnerMNIST

Main script to train a moth brain model on a crude (downsampled) MNIST set.
The moth can be generated from template or loaded complete from file.

Modifying parameters:
	1. Modify 'ModelParams' with the desired parameters for generating a moth
	(ie neural network), or specify a pre-existing 'model_params' file to load.
	2. Edit USER ENTRIES

The dataset:
	Because the moth brain architecture, as evolved, only handles ~60 features, we need to
create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
We do this by cropping and downsampling the mnist thumbnails, then selecting a subset of the
remaining pixels.
	This results in a cruder dataset (set various view flags to see thumbnails).
However, it is sufficient for testing the moth brain's learning ability. Other ML methods need
to be tested on this same cruder dataset to make useful comparisons.

Define train and control pools for the experiment, and determine the receptive field.
This is done first because the receptive field determines the number of AL units, which
     must be updated in model_params before 'initializeMatrixParams_fn' runs.
This dataset will be used for each simulation in numRuns. Each
     simulation draws a new set of samples from this set.

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

# Experiment details
from support_functions.generate import generate_ds_MNIST
from support_functions.show_figs import show_FA_thumbs, view_EN_resp
from support_functions.params import init_connection_matrix, ExpParams
from support_functions.sde import sde_wrap
from support_functions.classify import classify_digits_log_likelihood, classify_digits_thresholding

##TEST for Python vers == 3

### 1. Object initialization ###

## USER ENTRIES (Edit parameters below):
#-------------------------------------------------------------------------------
screen_size = (1920, 1080) # screen size (width, height)

use_existing_conn_matrices = False
# if True, load 'matrixParamsFilename', which includes filled-in connection matrices
# if False, generate new moth from template in params.py

matrix_params_filename = 'sampleMothModelParams'
# dict with all info, including connection matrices, of a particular moth

num_runs = 1 # how many runs you wish to do with this moth or moth template,
# each run using random draws from the mnist set

goal  = 15
# defines the moth's learning rates, in terms of how many training samples per
# class give max accuracy. So "goal = 1" gives a very fast learner.
# if goal == 0, the rate parameters defined the template will be used as-is
# if goal > 1, the rate parameters will be updated, even in a pre-set moth

tr_per_class = 3 # the number of training samples per class
num_sniffs = 2 # number of exposures each training sample

# nearest neighbors
run_nearest_neighbors = True # this option requires the sklearn library be installed
num_neighbors = 1 # optimization param for nearest neighbors
# Suggested values: tr_per_class ->
#	num_neighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

# SVM
runSVM = True # this option requires the sklearn library be installed
box_constraint = 1e1 # optimization parameter for svm
# Suggested values: tr_per_class ->
#	box_constraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1,
#					20 -> 1e-4 or 1e-5, 50 -> 1e-5 ; 100+ -> 1e-7

## Flags to show various images:
show_thumbnails = 0 # N means show N experiment inputs from each class
	# 0 means don't show any
show_EN_plots = [1, 1] # 1 to plot, 0 to ignore
# arg1 refers to statistical plots of EN response changes: One image (with 8 subplots) per EN
# arg2 refers to EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)

# To save results if wished:
save_all_neural_timecourses = False # 0 -> save only EN (ie readout) timecourses
# Caution: 1 -> very high memory demands, hinders longer runs
save_results_data = True
# save_results_image_data = True
results_filename = 'results' # will get the run number appended to it
save_results_data_folder = 'results/data' # String
# If `save_results_data` is True, 'results_filename' will be saved here
save_results_image_folder = 'results' # String
# If non-empty, results will be saved here (if show_EN_plots also non-zero)
save_params_folder = 'params' # String
# If non-empty, params will be saved here (if show_EN_plots also non-zero)

#-------------------------------------------------------------------------------

if run_nearest_neighbors or runSVM:
	pass
	##TEST to see if sklearn is installed,
	## if not, print message instructing to install it or
	## set run_nearest_neighbors and runSVM to False

if save_results_data: # or save_results_image_data:
	pass
	##TEST to see if dill is installed,
	## if not, print message instructing to install it or
	## set save_results_data to False
## Misc book-keeping
class_labels = np.array(range(10))  # For MNIST. '0' is labeled as 10
val_per_class = 15  # number of digits used in validation sets and in baseline sets

# make a vector of the classes of the training samples, randomly mixed:
tr_classes = np.repeat( class_labels, tr_per_class )
tr_classes = np.random.permutation( tr_classes )
# repeat these inputs if taking multiple sniffs of each training sample:
tr_classes = np.tile( tr_classes, [1, num_sniffs] )[0]

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

# Parameters:
# Parameters required for the dataset generation function are attached to a dictionary: 'preP'.
# 1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
#   This is NOT the number of training samples per class.
# 	That is tr_per_class, defined above.

# Specify pools of indices from which to draw baseline, train, val sets.
ind_pool_baseline = list(range(100)) # 1:100
ind_pool_train = list(range(100,300)) # 101:300
ind_pool_post = list(range(300,400)) # 301:400

## Create preP (preprocessingParams)
# Population preprocessing pools of indices:
preP = dict()
preP['indsToAverageGeneral'] = list(range(550,1000)) # 551:1000
preP['indsToCalculateReceptiveField'] = list(range(550,1000)) # 551:1000
preP['maxInd'] = max( [ preP['indsToCalculateReceptiveField'] + \
	ind_pool_train ][0] ) # we'll throw out unused samples

## 2. Pre-processing parameters for the thumbnails:
preP['screen_size'] = screen_size
preP['downsampleRate'] = 2
preP['crop'] = 2
preP['numFeatures'] =  85  # number of pixels in the receptive field
preP['pixelSum'] = 6
preP['showThumbnails'] = show_thumbnails # boolean
preP['downsampleMethod'] = 1 # 0 means sum square patches of pixels
							 # 1 means use bicubic interpolation

preP['classLabels'] = class_labels # append
preP['useExistingConnectionMatrices'] = use_existing_conn_matrices # boolean
preP['matrixParamsFilename'] = matrix_params_filename

# generate the data array:
fA, active_pixel_inds, len_side = generate_ds_MNIST(preP, saveImageFolder=save_results_image_folder)

_, num_per_class, class_num = fA.shape
# The dataset fA is a feature array ready for running experiments.
# Each experiment uses a random draw from this dataset.
# fA = n x m x 10 array where n = #active pixels, m = #digits from each class
# that will be used. The 3rd dimension gives the class, 1:10 where 10 = '0'.

#-------------------------------------------------------------------------------

### 3. Run MothNet simulation ###

# Loop through the number of simulations specified:
print(f'starting sim(s) for goal = {goal}, tr_per_class = {tr_per_class}, numSniffsPerSample = {num_sniffs}')

for run in range(num_runs):

	## Subsample the dataset for this simulation
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

	# show the final versions of thumbnails to be used, if wished
	if show_thumbnails:
		temp_array = np.zeros((len_side, num_per_class, class_num))
		temp_array[active_pixel_inds,:,:] = digit_queues
		normalize = 1
		show_FA_thumbs(temp_array, show_thumbnails, normalize, 'Input thumbnails',
							screen_size, save_results_image_folder)

#-------------------------------------------------------------------------------
	# Re-organize train and val sets for classifiers:

	# Build train and val feature matrices and class label vectors.
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

	# load an existing moth, or create a new moth
	if use_existing_conn_matrices:
		params_fname = os.path.join(save_params_folder, 'model_params.pkl')
		# load model_params
		with open(params_fname,'rb') as f:
			model_params = dill.load(f)
	else:
		# Load template params
		from support_functions.params import ModelParams
		model_params = ModelParams(nF=len(active_pixel_inds), goal=goal)

		# Now populate the moth's connection matrices using the model_params
		model_params = init_connection_matrix(model_params)

		# save params to file (if save_params_folder not empty)
		if save_params_folder:
			if not os.path.isdir(save_params_folder):
				os.mkdir(save_params_folder)
				# pickle parameters for other branch of if construct
				params_fname = os.path.join(save_params_folder, 'model_params.pkl')
				dill.dump(model_params, open(params_fname, 'wb'))

	model_params.trueClassLabels = class_labels # misc parameter tagging along
	model_params.saveAllNeuralTimecourses = save_all_neural_timecourses

	# # Define the experiment parameters, including book-keeping for time-stepped
	# # 	evolutions, eg when octopamine occurs, time regions to poll for digit
	# # 	responses, windowing of firing rates, etc
	# Load experiment params, including book-keeping for time-stepped
	# 	evolutions, eg when octopamine occurs, time regions to poll for digit
	# 	responses, windowing of Firing rates, etc
	experiment_params = ExpParams(tr_classes, class_labels, val_per_class)

#-------------------------------------------------------------------------------

	# 3. run this experiment as sde time-step evolution:
	sim_results = sde_wrap( model_params, experiment_params, digit_queues )

#-------------------------------------------------------------------------------

	# Experiment Results: EN behavior, classifier calculations:
	if save_results_image_folder:
		if not os.path.isdir(save_results_image_folder):
			os.mkdir(save_results_image_folder)

	# Process the sim results to group EN responses by class and time:
	resp_orig = view_EN_resp(sim_results, model_params, experiment_params,
		show_EN_plots, class_labels, screen_size, results_filename, save_results_image_folder)

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
	print( 'LogLikelihood:' )
	print( f"Naive Accuracy: {round(output_naive_log_loss['total_acc'])}" + \
		f"#, by class: {np.round(output_naive_log_loss['acc_perc'])} #.   ")

	# Post-training accuracy using log-likelihood over all ENs:
	output_trained_log_loss = classify_digits_log_likelihood( resp_orig )
	print( f"Trained Accuracy: {round(output_trained_log_loss['total_acc'])}" + \
		f"#, by class: {np.round(output_trained_log_loss['acc_perc'])} #.   ")

	# 2. Using single EN thresholding:
	output_naive_thresholding = classify_digits_thresholding( resp_naive, 1e9, -1, 10 )
	output_trained_thresholding = classify_digits_thresholding( resp_orig, 1e9, -1, 10 )

	# append the accuracy results, and other run data, to the first entry of resp_orig:
	resp_orig[0]['modelParams'] = model_params  # will include all connection weights of this moth
	resp_orig[0]['outputNaiveLogL'] = output_naive_log_loss
	resp_orig[0]['outputTrainedLogL'] = output_trained_log_loss
	resp_orig[0]['outputNaiveThresholding'] = output_naive_thresholding
	resp_orig[0]['outputTrainedThresholding'] = output_trained_thresholding
	resp_orig[0]['matrixParamsFilename'] = matrix_params_filename
	resp_orig[0]['K2Efinal'] = sim_results['K2Efinal']

	if save_results_data_folder and save_results_data:
		if not os.path.isdir(save_results_data_folder):
			os.mkdir(save_results_data_folder)
			print(f"Creating results directory:\n{save_results_data_folder}")

		import dill # for pickling module object (optional)

		# save results data
		results_fname = os.path.join(save_results_data_folder, f'{results_filename}_{run}.pkl')
		dill.dump(resp_orig, open(results_fname, 'wb'))
		# open via:
		# >>> with open(results_fname,'rb') as f:
    	# >>> 	B = dill.load(f)

		print(f'Results saved to: {results_fname}')

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
		# old acc. metric: np.sum(y_hat == val_y.squeeze()) / len(val_y)

		# measure accuracy for each class
		class_acc = np.zeros_like(class_labels) # preallocate
		for i in class_labels:
			inds = np.where(val_y==i)[0]
			class_acc[i] = np.round(100*np.sum( y_hat[inds]==val_y[inds].squeeze()) /
				len(val_y[inds]) )

		print( f'Nearest neighbor: {tr_per_class} training samples per class.',
            f' Accuracy = {np.round(100*nn_acc)}%. numNeigh = {num_neighbors}.',
            f' Class accs (%): {class_acc}' )

#-------------------------------------------------------------------------------

	# support vector machine
	if runSVM:

		from sklearn import svm
		svm_clf = svm.SVC(gamma='scale', C=box_constraint)
		svm_clf.fit(train_X, train_y.ravel())
		y_hat = svm_clf.predict(val_X)

		# measure overall accuracy
		svm_acc = svm_clf.score(val_X, val_y)
		# old acc. metric: np.sum(y_hat == val_y.squeeze()) / len(val_y)

		# measure accuracy for each class
		class_acc = np.zeros_like(class_labels) # preallocate
		for i in class_labels:
			inds = np.where(val_y==i)[0]
			class_acc[i] = np.round(100*np.sum( y_hat[inds]==val_y[inds].squeeze()) /
				len(val_y[inds]) )

		print( f'Support vector machine: {tr_per_class} training samples per class.',
            f' Accuracy = {np.round(100*svm_acc)}%. BoxConstraint(i.e. C) = {box_constraint}.',
            f' Class accs (%): {class_acc}' )

print('         -------------All done-------------         ')

runDuration = time.time() - runStart
print(f'{__file__} executed in {runDuration/60:.3f} minutes')

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
