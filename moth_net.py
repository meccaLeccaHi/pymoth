'''
moth_net

Python module to train a moth brain model on a crude (downsampled) MNIST set.
The moth can be generated from template or loaded complete from file.
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
import numpy as np
import os
import copy # for deep copy of nested lists
import sys

# Experiment details
from support_functions.generate import generate_ds_MNIST
from support_functions.show_figs import show_FA_thumbs, show_EN_resp
from support_functions.params import init_connection_matrix, ExpParams
from support_functions.sde import sde_wrap
from support_functions.classify import classify_digits_log_likelihood, classify_digits_thresholding


##TEST for Python version > 2
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if sys.version_info.major > 2:
	print(f"Python version {python_version} detected. ")
else:
	version_error = f"Python version {python_version} detected.\n" + \
					f"Python version 3 or higher is required to run this module.\n" + \
					"Please install Python 3+."
	raise Exception(version_error)

#-------------------------------------------------------------------------------
## User-editable parameters:

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

class MothNet:
    """
    This is a class for mathematical operations on complex numbers.

    Attributes:
        real (int): The real part of complex number.
        imag (int): The imaginary part of complex number.
    """

    # Class Attribute
    species = 'mammal'

    # Initializer / Instance Attributes
    def __init__(self, name, age):
        """
        The constructor for MothNet class.

        Parameters:
           name (int): The real part of complex number.
           age (int): The imaginary part of complex number.
        """
        self.name = name
        self.age = age

    # instance method
    def description(self):
        """
        Description of this particular moth.

        Returns:
            String: A string describing this particular moth.
        """
        return "{} is {} years old".format(self.name, self.age)

    # instance method
    def fit(self, X, y):
        """
        Fit moth using training samples.

        Parameters:
            X (array): Feature matrix (m x n)
            y (array): Label vector
        """

        runStart = time.time() # time execution duration

        # some code

        runDuration = time.time() - runStart
        print(f'{__file__} executed in {runDuration/60:.3f} minutes')

        return "{} says {}".format(self.name, sound)


##MOVE THIS TO ANOTHER SCRIPT

# from moth_net import MothNet

# Instantiate the Dog object
mothra = MothNet("Mikey", 6)

# call our instance methods
print(mothra.description())




# ##IDEAL USAGE
# from moth_net import MothNet
#
# # Load MNIST dataset
# X_vals, y_vals = moth_net.load_MNIST()
#
# # Instantiate the MothNet object
# mothra = moth_net.MothNet(moth_parameters, experiment_parameters)
#
# mothra.fit_on_MNIST()
# # mothra.fit(X_train, y_train)
#
# mnist_accuracy = mothra.score_on_MNIST()
# # moth_accuracy = mothra.score(X_test, y_test)
