#!/usr/bin/env python3

print("\nWARNING: This package is still under development.")
print("Use procedural version by running `$ python runMothLearnerMNIST.py` from the parent directory.\n")

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
    print("Python version {} detected.\n".format(python_version))
else:
    version_error = "Python version {} detected.\n".format(python_version) + \
                    "Python version 3 or higher is required to run this module.\n" + \
                    "Please install Python 3+."
    raise Exception(version_error)

class MothNet:
    '''
    Python module to train a moth brain model on a crude (downsampled) MNIST set.
    The moth can be generated from template or loaded complete from file.
    Main script to train a moth brain model on a crude (downsampled) MNIST set.
    The moth can be generated from template or loaded complete from file.

    Modifying parameters:
    	1. Modify 'ModelParams' with the desired parameters for generating a moth
        (ie neural network)
    	2. Edit USER ENTRIES

    The dataset:
    	Because the moth brain architecture, as evolved, only handles ~60 features,
    we need to create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
    We do this by cropping and downsampling the mnist thumbnails, then selecting
    a subset of the remaining pixels.
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
    	3. Create a moth (neural net). Either select an existing moth file, or generate
            a new moth in 2 steps:
    		a) run 'ModelParams' and incorporate user entry edits such as 'GOAL'
    		b) create connection matrices via 'init_connection_matrix'
    	4. Load the experiment parameters
    	5. Run the simulation with 'sde_wrap', print results to console
    	6. Plot results (optional)
    	7. Run addition ML models for comparison, print results to console (optional)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    # Experiment details
    from modules.sde import collect_stats

    # Initializer / Instance Attributes
    def __init__(self):
        """
        The constructor for MothNet class.
        """

        ## USER ENTRIES (constants)
        # Edit parameters below this line:
        #-------------------------------------------------------------------------------
        self.SCREEN_SIZE = (1920, 1080) # screen size (width, height)

        # use_existing_conn_matrices = False
        ## if True, load 'matrixParamsFilename', which includes filled-in connection matrices
        ## if False, generate new moth from template in params.py

        # matrix_params_filename = 'sampleMothModelParams'
        ## dict with all info, including connection matrices, of a particular moth

        self.NUM_RUNS = 1 # how many runs you wish to do with this moth or moth template,
        # each run using random draws from the mnist set

        self.GOAL  = 15
        # defines the moth's learning rates, in terms of how many training samples per
        # class give max accuracy. So "GOAL = 1" gives a very fast learner.
        # if GOAL == 0, the rate parameters defined the template will be used as-is
        # if GOAL > 1, the rate parameters will be updated, even in a pre-set moth

        self.TR_PER_CLASS = 1 # (try 3) the number of training samples per class
        self.NUM_SNIFFS = 1 # (try 2) number of exposures each training sample

        # nearest neighbors
        self.RUN_NEAREST_NEIGHBORS = True # this option requires the sklearn library be installed
        self.NUM_NEIGHBORS = 1 # optimization param for nearest neighbors
        # Suggested values: TR_PER_CLASS ->
        #	NUM_NEIGHBORS:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

        # SVM
        self.RUN_SVM = True # this option requires the sklearn library be installed
        self.BOX_CONSTRAINT = 1e1 # optimization parameter for svm
        # Suggested values: TR_PER_CLASS ->
        #	BOX_CONSTRAINT:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1,
        #					20 -> 1e-4 or 1e-5, 50 -> 1e-5 ; 100+ -> 1e-7

        ## Flags to show various images:
        self.N_THUMBNAILS = 1 # N means show N experiment inputs from each class
        	# 0 means don't show any

        # flag for statistical plots of EN response changes: One image (with 8 subplots) per EN
        self.SHOW_ACC_PLOTS = True # True to plot, False to ignore
        # flag for EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)
        self.SHOW_TIME_PLOTS = True # True to plot, False to ignore
        # flag for ROC multi-class ROC curves (one for each model)
        self.SHOW_ROC_PLOTS = True # True to plot, False to ignore

        self.RESULTS_FOLDER = os.getcwd() + os.sep + 'results' # string (absolute path)
        # If non-empty, results will be saved here
        self.RESULTS_FILENAME = 'results' # will get the run number appended to it

        #-------------------------------------------------------------------------------

        # Test parameters for compatibility
        if self.RUN_NEAREST_NEIGHBORS or self.RUN_SVM:
        	##TEST to see if sklearn is installed,
        	try:
        	    import sklearn
        	except ImportError:
        	    print('sklearn is not installed, and it is required to run ML models.\n' + \
        			"Install it or set RUN_NEAREST_NEIGHBORS and RUN_SVM to 'False'.")

        if self.SHOW_ACC_PLOTS or self.SHOW_TIME_PLOTS:
        	##TEST that directory string is not empty
        	if not self.RESULTS_FOLDER:
        		folder_error = "RESULTS_FOLDER parameter is empty.\n" + \
        			"Please add directory or set SHOW_ACC_PLOTS and SHOW_TIME_PLOTS to 'False'."
        		raise Exception(folder_error)

        	##TEST for existence of image results folder, else create it
        	if not os.path.isdir(self.RESULTS_FOLDER):
        		os.mkdir(self.RESULTS_FOLDER)
        		print('Creating results directory: {}'.format(self.RESULTS_FOLDER))

    ### 2. Load and preprocess MNIST dataset ###

    def load_MNIST(self):
        '''
        Load and preprocess MNIST dataset
        The dataset:
        Because the moth brain architecture, as evolved, only handles ~60 features, we need to
        create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
        We do this by cropping and downsampling the MNIST thumbnails, then selecting
        a subset of the remaining pixels.
        This results in a cruder dataset (set various view flags to see thumbnails).
        However, it is sufficient for testing the moth brain's learning ability. Other
        ML methods need to be tested on this same cruder dataset to make useful comparisons.

        Define train and control pools for the experiment, and determine the receptive field.
        This is done first because the receptive field determines the number of AL units, which
             must be updated in model_params before 'initializeMatrixParams_fn' runs.
        This dataset will be used for each simulation in NUM_RUNS. Each
             simulation draws a new set of samples from this set.

        Parameters required for the dataset generation function:
        1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
          This is NOT the number of training samples per class.
        	That is TR_PER_CLASS, defined above.
        '''

        from modules.generate import generate_ds_MNIST

        self._class_labels = np.array(range(10))  # For MNIST. '0' is labeled as 10
        self._val_per_class = 15  # number of digits used in validation sets and in baseline sets

        # make a vector of the classes of the training samples, randomly mixed:
        self._tr_classes = np.repeat( self._class_labels, self.TR_PER_CLASS )
        self._tr_classes = np.random.permutation( self._tr_classes )
        # repeat these inputs if taking multiple sniffs of each training sample:
        self._tr_classes = np.tile( self._tr_classes, [1, self.NUM_SNIFFS] )[0]

        # Specify pools of indices from which to draw baseline, train, val sets.
        self._ind_pool_baseline = list(range(100)) # 1:100
        self._ind_pool_train = list(range(100,300)) # 101:300
        self._ind_pool_post = list(range(300,400)) # 301:400

        ## Create preprocessing parameters
        # Population pre-processing pools of indices:
        self._inds_to_ave = list(range(550,1000))
        self._inds_to_calc_RF = list(range(550,1000))
        self._max_ind = max( [ self._inds_to_calc_RF + self._ind_pool_train ][0] ) # we'll throw out unused samples

        ## 2. Pre-processing parameters for the thumbnails:
        self._downsample_rate = 2
        self._crop = 2
        self._num_features = 85 # number of pixels in the receptive field
        self._pixel_sum = 6
        self._show_thumbnails = self.N_THUMBNAILS
        self._downsample_method = 1 # 0 means sum square patches of pixels
        					# 1 means use bicubic interpolation

        # generate the data array:
        # _feat_array is a feature array ready for running experiments.
        # Each experiment uses a random draw from this dataset.
        self._feat_array, self._active_pixel_inds, self._len_side = generate_ds_MNIST(
        	self._max_ind, self._class_labels, self._crop, self._downsample_rate,
            self._downsample_method, self._inds_to_ave, self._pixel_sum,
            self._inds_to_calc_RF, self._num_features, self.SCREEN_SIZE,
            self.RESULTS_FOLDER, self._show_thumbnails
        	)

        _, self._num_per_class, self._class_num = self._feat_array.shape
        # _feat_array = n x m x 10 array where n = #active pixels, m = #digits from each class
        # that will be used. The 3rd dimension gives the class: 0:9.

        # Line up the images for the experiment (in 10 parallel queues)
        digit_queues = np.zeros_like(self._feat_array)

        for i in self._class_labels:

            ## 1. Baseline (pre-train) images
            # choose some images from the baselineIndPool
            _range_top_end = max(self._ind_pool_baseline) - min(self._ind_pool_baseline) + 1
            _r_sample = np.random.choice(_range_top_end, self._val_per_class) # select random digits
            _these_inds = min(self._ind_pool_baseline) + _r_sample
            digit_queues[:,:self._val_per_class,i] = self._feat_array[:,_these_inds,i]

            ## 2. Training images
            # choose some images from the trainingIndPool
            _range_top_end = max(self._ind_pool_train) - min(self._ind_pool_train) + 1
            _r_sample = np.random.choice(_range_top_end, self.TR_PER_CLASS) # select random digits
            _these_inds = min(self._ind_pool_train) + _r_sample
            # repeat these inputs if taking multiple sniffs of each training sample
            _these_inds = np.tile(_these_inds, self.NUM_SNIFFS)
            digit_queues[:, self._val_per_class:(self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS), i] = \
                self._feat_array[:, _these_inds, i]

            ## 3. Post-training (val) images
            # choose some images from the postTrainIndPool
            _range_top_end = max(self._ind_pool_post) - min(self._ind_pool_post) + 1
            _r_sample = np.random.choice(_range_top_end, self._val_per_class) # select random digits
            _these_inds = min(self._ind_pool_post) + _r_sample
            digit_queues[:,(self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS): \
                (self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS+self._val_per_class),
    			i] = self._feat_array[:, _these_inds, i]

        # show the final versions of thumbnails to be used, if wished
        if self.N_THUMBNAILS:
            from modules.show_figs import show_FA_thumbs
            _thumb_array = np.zeros((self._len_side, self._num_per_class, self._class_num))
            _thumb_array[self._active_pixel_inds,:,:] = digit_queues
            normalize = 1
            show_FA_thumbs(_thumb_array, self.N_THUMBNAILS, normalize, 'Input thumbnails',
    			self.SCREEN_SIZE, self.RESULTS_FOLDER + os.sep + 'thumbnails')

        return digit_queues

#-------------------------------------------------------------------------------

    def train_test_split(self, digit_queues):
        '''
        Subsample the dataset for this simulation,
        then build train and val feature matrices and class label vectors.

        Returns: train_X, val_X, train_y, val_y
        '''

        # X = n x numberPixels;  Y = n x 1, where n = 10*TR_PER_CLASS.
        train_X = np.zeros((10*self.TR_PER_CLASS, self._feat_array.shape[0]))
        val_X = np.zeros((10*self._val_per_class, self._feat_array.shape[0]))
        train_y = np.zeros((10*self.TR_PER_CLASS, 1))
        val_y = np.zeros((10*self._val_per_class, 1))

        # populate the labels one class at a time
        for i in self._class_labels:
            # skip the first '_val_per_class' digits,
            # as these are used as baseline digits in the moth (formality)
            temp = digit_queues[:,self._val_per_class:self._val_per_class+self.TR_PER_CLASS,i]
            train_X[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS,:] = temp.T
            temp = digit_queues[:,self._val_per_class+self.TR_PER_CLASS: \
                2*self._val_per_class+self.TR_PER_CLASS,i]
            val_X[i*self._val_per_class:(i+1)*self._val_per_class,:] = temp.T
            train_y[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS] = i
            val_y[i*self._val_per_class:(i+1)*self._val_per_class,:] = i

        return train_X, val_X, train_y, val_y

    def load_moth(self):
        '''
        Create a new moth, ie the template that is used to populate connection
            matrices and to control behavior.
        Returns: Model parameters
        '''
        from modules.params import ModelParams as ModelParams

    	# instantiate template params
        model_params = ModelParams( len(self._active_pixel_inds), self.GOAL )

    	# populate the moth's connection matrices using the model_params
        model_params.init_connection_matrix()

        return model_params

    def load_exp(self):
        '''
        Load experiment parameters, including book-keeping for time-stepped
    	   evolutions, eg when octopamine occurs, time regions to poll for digit
           responses, windowing of firing rates, etc.
        Returns: Experiment parameters
        '''

        from modules.params import ExpParams

        return ExpParams( self._tr_classes, self._class_labels, self._val_per_class )

    def simulate(self, model_params, experiment_params, digit_queues):
        '''
        Runs the SDE time-stepped evolution of neural firing rates.
        Parameters:
            1. model_params: object with connection matrices etc
            2. exp_params: object with timing info about experiment, eg when stimuli are given.
            3. feature_array: array of stimuli (numFeatures x numStimsPerClass x numClasses)
        Returns:
            1. sim_results: EN timecourses and final P2K and K2E connection matrices.
              Note that other neurons' timecourses (outputted from sdeEvolutionMnist)
              are not retained in sim_results.

        #-----------------------------------------------------------------------

        4 sections:
            - load various params needed for pre-evolution prep
            - specify stim and octo courses
            - interaction equations and step through simulation
            - unpack evolution output and export

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''
        from modules.sde import sde_wrap

        print('\nStarting sim for goal = {}, tr_per_class = {}, numSniffsPerSample = {}'.format(
    		self.GOAL, self.TR_PER_CLASS, self.NUM_SNIFFS))

        # run this experiment as sde time-step evolution:
        return sde_wrap( model_params, experiment_params, digit_queues )

    def score_moth_on_MNIST(self, EN_resp_trained):
        '''
        Calculate the classification accuracy of MothNet on MNIST dataset.
        Prints:
            1. output_naive_log_loss: Baseline accuracy using log-likelihoods over all ENs
            1. output_trained_log_loss: Post-training accuracy using log-likelihoods over all ENs
            1. output_naive_thresholding: Baseline accuracy using single EN thresholding
            1. output_trained_thresholding: Baseline accuracy using single EN thresholding
        Returns:
            None

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''
        from modules.classify import classify_digits_log_likelihood, classify_digits_thresholding
        from modules.show_figs import show_roc_curves

        # for baseline accuracy function argin, substitute pre- for post-values in EN_resp_trained:
        EN_resp_naive = copy.deepcopy(EN_resp_trained)
        for i, resp in enumerate(EN_resp_trained):
            EN_resp_naive[i]['post_mean_resp'] = resp['pre_mean_resp'].copy()
            EN_resp_naive[i]['post_std_resp'] = resp['pre_std_resp'].copy()
            EN_resp_naive[i]['post_train_resp'] = resp['pre_train_resp'].copy()

        # 1. using log-likelihoods over all ENs:
        # baseline accuracy:
        output_naive_log_loss = classify_digits_log_likelihood( EN_resp_naive )
        # post-training accuracy using log-likelihood over all ENs:
        output_trained_log_loss = classify_digits_log_likelihood( EN_resp_trained )

        print('LogLikelihood:')
        print(' Baseline (Naive) Accuracy: {}%,'.format(round(output_naive_log_loss['total_acc'])) + \
    		'by class: {}%'.format(np.round(output_naive_log_loss['acc_perc'])))
        print(' Trained Accuracy: {}%,'.format(round(output_trained_log_loss['total_acc'])) + \
    		'by class: {}%'.format(np.round(output_trained_log_loss['acc_perc'])))

        # 2. using single EN thresholding:
        output_naive_thresholding = classify_digits_thresholding( EN_resp_naive, 1e9, -1, 10 )
        output_trained_thresholding = classify_digits_thresholding( EN_resp_trained, 1e9, -1, 10 )

        print('Thresholding:')
        print(' Baseline (Naive) Accuracy: {}%,'.format(round(output_naive_thresholding['total_acc'])) + \
    		'by class: {}%'.format(np.round(output_naive_thresholding['acc_perc'])))
        print(' Trained Accuracy: {}%,'.format(round(output_trained_thresholding['total_acc'])) + \
    		'by class: {}%'.format(np.round(output_trained_thresholding['acc_perc'])))

        if self.SHOW_ROC_PLOTS:
            # compute macro-average ROC curve
            show_roc_curves(output_trained_log_loss['fpr'], output_trained_log_loss['tpr'],
        		output_trained_log_loss['roc_auc'], self._class_labels,
        		title_str='MothNet',
                images_filename=self.RESULTS_FOLDER + os.sep + self.RESULTS_FILENAME)
