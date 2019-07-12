class MothNet:
    '''
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
    from support_functions.show_figs import show_FA_thumbs, show_EN_resp
    from support_functions.params import ExpParams
    from support_functions.sde import sde_wrap
    from support_functions.classify import classify_digits_log_likelihood, classify_digits_thresholding

    # Initializer / Instance Attributes
    def __init__(self):
        """
        The constructor for MothNet class.
        """

        ## USER ENTRIES (Edit parameters below):
        #-------------------------------------------------------------------------------
        self.screen_size = (1920, 1080) # screen size (width, height)

        # use_existing_conn_matrices = False
        ## if True, load 'matrixParamsFilename', which includes filled-in connection matrices
        ## if False, generate new moth from template in params.py

        # matrix_params_filename = 'sampleMothModelParams'
        ## dict with all info, including connection matrices, of a particular moth

        self.num_runs = 1 # how many runs you wish to do with this moth or moth template,
        # each run using random draws from the mnist set

        self.goal  = 15
        # defines the moth's learning rates, in terms of how many training samples per
        # class give max accuracy. So "goal = 1" gives a very fast learner.
        # if goal == 0, the rate parameters defined the template will be used as-is
        # if goal > 1, the rate parameters will be updated, even in a pre-set moth

        self.tr_per_class = 3 # the number of training samples per class
        self.num_sniffs = 2 # number of exposures each training sample

        # nearest neighbors
        self.run_nearest_neighbors = True # this option requires the sklearn library be installed
        self.num_neighbors = 1 # optimization param for nearest neighbors
        # Suggested values: tr_per_class ->
        #	num_neighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

        # SVM
        self.runSVM = True # this option requires the sklearn library be installed
        self.box_constraint = 1e1 # optimization parameter for svm
        # Suggested values: tr_per_class ->
        #	box_constraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1,
        #					20 -> 1e-4 or 1e-5, 50 -> 1e-5 ; 100+ -> 1e-7

        ## Flags to show various images:
        self.n_thumbnails = 1 # N means show N experiment inputs from each class
        	# 0 means don't show any

        # To save results if wished:
        self.save_all_neural_timecourses = False # 0 -> save only EN (ie readout) timecourses
        # Caution: 1 -> very high memory demands, hinders longer runs

        # flag for statistical plots of EN response changes: One image (with 8 subplots) per EN
        self.show_acc_plots = True # True to plot, False to ignore
        # flag for EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)
        self.show_time_plots = True # True to plot, False to ignore
        self.save_results_folder = 'results' # String (relative path)
        # If non-empty, results will be saved here

        self.results_filename = 'results' # will get the run number appended to it

        #-------------------------------------------------------------------------------

        # Test parameters for compatibility
        if self.run_nearest_neighbors or self.runSVM:
        	##TEST to see if sklearn is installed,
        	try:
        	    import sklearn
        	except ImportError:
        	    print('sklearn is not installed, and it is required to run ML models.\n' + \
        			"Install it or set run_nearest_neighbors and runSVM to 'False'.")

        if self.show_acc_plots or self.show_time_plots:
        	##TEST that directory string is not empty
        	if not self.save_results_folder:
        		folder_error = "save_results_folder parameter is empty.\n" + \
        			"Please add directory or set show_acc_plots and show_time_plots to 'False'."
        		raise Exception(folder_error)

        	##TEST for existence of image results folder, else create it
        	if not self.os.path.isdir(self.save_results_folder):
        		self.os.mkdir('./'+self.save_results_folder)
        		print('Creating results directory: {}'.format(
                    self.os.path.join(self.os.getcwd(),self.save_results_folder)))

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
        This dataset will be used for each simulation in num_runs. Each
             simulation draws a new set of samples from this set.

        Parameters required for the dataset generation function:
        1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
          This is NOT the number of training samples per class.
        	That is tr_per_class, defined above.
        '''

        self.class_labels = self.np.array(range(10))  # For MNIST. '0' is labeled as 10
        self.val_per_class = 15  # number of digits used in validation sets and in baseline sets

        # make a vector of the classes of the training samples, randomly mixed:
        self.tr_classes = self.np.repeat( self.class_labels, self.tr_per_class )
        self.tr_classes = self.np.random.permutation( self.tr_classes )
        # repeat these inputs if taking multiple sniffs of each training sample:
        self.tr_classes = self.np.tile( self.tr_classes, [1, self.num_sniffs] )[0]

        # Specify pools of indices from which to draw baseline, train, val sets.
        self.ind_pool_baseline = list(range(100)) # 1:100
        self.ind_pool_train = list(range(100,300)) # 101:300
        self.ind_pool_post = list(range(300,400)) # 301:400

        ## Create preprocessing parameters
        # Population pre-processing pools of indices:
        self.inds_to_ave = list(range(550,1000))
        self.inds_to_calc_RF = list(range(550,1000))
        self.max_ind = max( [ self.inds_to_calc_RF + self.ind_pool_train ][0] ) # we'll throw out unused samples

        ## 2. Pre-processing parameters for the thumbnails:
        self.downsample_rate = 2
        self.crop = 2
        self.num_features = 85 # number of pixels in the receptive field
        self.pixel_sum = 6
        self.show_thumbnails = self.n_thumbnails
        self.downsample_method = 1 # 0 means sum square patches of pixels
        					# 1 means use bicubic interpolation

        # generate the data array:
        # The dataset fA is a feature array ready for running experiments.
        # Each experiment uses a random draw from this dataset.
        self.fA, self.active_pixel_inds, self.len_side = self.generate_ds_MNIST(
        	self.max_ind, self.class_labels, self.crop, self.downsample_rate,
            self.downsample_method, self.inds_to_ave, self.pixel_sum,
            self.inds_to_calc_RF, self.num_features, self.screen_size,
            self.save_results_folder, self.show_thumbnails
        	)

        _, self.num_per_class, self.class_num = self.fA.shape
        # fA = n x m x 10 array where n = #active pixels, m = #digits from each class
        # that will be used. The 3rd dimension gives the class: 0:9.

#-------------------------------------------------------------------------------

    def train_test_split(self):
        '''
        Subsample the dataset for this simulation,
        then build train and val feature matrices and class label vectors.

        Returns: train_X, val_X, train_y, val_y
        '''

        ##ADD TEST for presence of self.fA,
        # ELSE print warning to load data first

        # Line up the images for the experiment (in 10 parallel queues)
        digit_queues = self.np.zeros_like(self.fA)

        for i in self.class_labels:

            ## 1. Baseline (pre-train) images
            # choose some images from the baselineIndPool
            range_top_end = max(self.ind_pool_baseline) - min(self.ind_pool_baseline) + 1
            r_sample = self.np.random.choice(range_top_end, self.val_per_class) # select random digits
            these_inds = min(self.ind_pool_baseline) + r_sample
            digit_queues[:,:self.val_per_class,i] = self.fA[:,these_inds,i]

            ## 2. Training images
            # choose some images from the trainingIndPool
            range_top_end = max(self.ind_pool_train) - min(self.ind_pool_train) + 1
            r_sample = self.np.random.choice(range_top_end, self.tr_per_class) # select random digits
            these_inds = min(self.ind_pool_train) + r_sample
            # repeat these inputs if taking multiple sniffs of each training sample
            these_inds = self.np.tile(these_inds, self.num_sniffs)
            digit_queues[:, self.val_per_class:(self.val_per_class+self.tr_per_class*self.num_sniffs), i] = \
                self.fA[:, these_inds, i]

            ## 3. Post-training (val) images
            # choose some images from the postTrainIndPool
            range_top_end = max(self.ind_pool_post) - min(self.ind_pool_post) + 1
            r_sample = self.np.random.choice(range_top_end, self.val_per_class) # select random digits
            these_inds = min(self.ind_pool_post) + r_sample
            digit_queues[:,(self.val_per_class+self.tr_per_class*self.num_sniffs): \
                (self.val_per_class+self.tr_per_class*self.num_sniffs+self.val_per_class),
    			i] = self.fA[:, these_inds, i]

        # X = n x numberPixels;  Y = n x 1, where n = 10*tr_per_class.
        train_X = self.np.zeros((10*self.tr_per_class, self.fA.shape[0]))
        val_X = self.np.zeros((10*self.val_per_class, self.fA.shape[0]))
        train_y = self.np.zeros((10*self.tr_per_class, 1))
        val_y = self.np.zeros((10*self.val_per_class, 1))

        # populate the labels one class at a time
        for i in self.class_labels:
            # skip the first 'val_per_class' digits,
            # as these are used as baseline digits in the moth (formality)
            temp = digit_queues[:,self.val_per_class:self.val_per_class+self.tr_per_class,i]
            train_X[i*self.tr_per_class:(i+1)*self.tr_per_class,:] = temp.T
            temp = digit_queues[:,self.val_per_class+self.tr_per_class: \
                2*self.val_per_class+self.tr_per_class,i]
            val_X[i*self.val_per_class:(i+1)*self.val_per_class,:] = temp.T
            train_y[i*self.tr_per_class:(i+1)*self.tr_per_class] = i
            val_y[i*self.val_per_class:(i+1)*self.val_per_class,:] = i

        return train_X, val_X, train_y, val_y

    def load_moth(self):

        from support_functions.params import ModelParams

        ## Create a new moth:
    	# instantiate template params
        model_params = ModelParams( len(self.active_pixel_inds), self.goal )

    	# Populate the moth's connection matrices using the model_params
        model_params.init_connection_matrix()

        return model_params

    # # instance method
    # def description(self):
    #     """
    #     Description of this particular moth.
    #
    #     Returns:
    #         String: A string describing this particular moth.
    #     """
    #     return "{} is {} years old".format(self.name, self.age)
    #
    # # instance method
    # def fit(self, X, y):
    #     """
    #     Fit moth using training samples.
    #
    #     Parameters:
    #         X (array): Feature matrix (m x n)
    #         y (array): Label vector
    #     """
    #
    #     runStart = time.time() # time execution duration
    #
    #     # some code
    #
    #     runDuration = time.time() - runStart
    #     print(f'{__file__} executed in {runDuration/60:.3f} minutes')
    #
    #     return "{} says {}".format(self.name, sound)


# ##MOVE THIS TO ANOTHER SCRIPT
#
# # from moth_net import MothNet
#
# # Instantiate the MothNet object
# mothra = MothNet()
#
# # call our instance methods
# mothra.load_MNIST()
# train_X, val_X, train_y, val_y = mothra.train_test_split()
#
# # Load parameters
# moth_parameters = mothra.load_moth() # define moth model parameters
# experiment_parameters = mothra.load_exp() # define parameters of a time-evolution experiment
#
# mothra.fit_on_MNIST()
# # mothra.fit(X_train, y_train)
#
# mnist_accuracy = mothra.score_on_MNIST()
# svm_accuracy = mothra.score_svm(X_test, y_test)
# knn_accuracy = mothra.score_knn(X_test, y_test)
