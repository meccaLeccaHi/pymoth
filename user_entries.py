#!/usr/bin/env python3

## USER ENTRIES (constants)
# Edit parameters below this line:
#-------------------------------------------------------------------------------
SCREEN_SIZE = (1920, 1080) # screen size (width, height)

NUM_RUNS = 1 # how many runs you wish to do with this moth or moth template,
# each run using random draws from the mnist set

GOAL = 15
# defines the moth's learning rates, in terms of how many training samples per
# class give max accuracy. So "GOAL = 1" gives a very fast learner.
# if GOAL == 0, the rate parameters defined the template will be used as-is
# if GOAL > 1, the rate parameters will be updated, even in a pre-set moth

TR_PER_CLASS = 1 # (try 3) the number of training samples per class
NUM_SNIFFS = 1 # (try 2) number of exposures each training sample

# nearest neighbors
NUM_NEIGHBORS = 1 # optimization param for nearest neighbors
# Suggested values:
#	NUM_NEIGHBORS:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

# SVM
BOX_CONSTRAINT = 1e1 # optimization parameter for svm
# Suggested values:
#	BOX_CONSTRAINT:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1,
#					20 -> 1e-4 or 1e-5, 50 -> 1e-5 ; 100+ -> 1e-7

## Flags to show various images:
N_THUMBNAILS = 1 # show N experiment inputs from each class
  # 0 means don't show any

# flag for statistical plots of EN response changes: One image (with 8 subplots) per EN
SHOW_ACC_PLOTS = True # True to plot, False to ignore
# flag for EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)
SHOW_TIME_PLOTS = True # True to plot, False to ignore
# flag for ROC multi-class ROC curves (one for each model)
SHOW_ROC_PLOTS = True # True to plot, False to ignore

RESULTS_FOLDER = 'results' # string
# (relative path, starting inside the directory housing this package)
# If non-empty, results will be saved there
RESULTS_FILENAME = 'results' # will get the run number appended to it
