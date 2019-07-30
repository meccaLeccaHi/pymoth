#!/usr/bin/env python3

print("\nWARNING: This package is still under development.")
print("Use procedural version by running `$ python runMothLearnerMNIST.py` from the parent directory.\n")

# import packages
import numpy as _np
import os as _os
import copy as _copy # for deep copy of nested lists
import sys as _sys

##TEST for Python version > 2
_python_version = "{}.{}".format(_sys.version_info.major,_sys.version_info.minor)
if _sys.version_info.major > 2:
    print("Python version {} detected.\n".format(_python_version))
else:
    version_error = "Python version {} detected.\n".format(_python_version) + \
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

        self.NUM_RUNS = 1 # how many runs you wish to do with this moth or moth template,
        # each run using random draws from the mnist set

        self.GOAL = 15
        # defines the moth's learning rates, in terms of how many training samples per
        # class give max accuracy. So "GOAL = 1" gives a very fast learner.
        # if GOAL == 0, the rate parameters defined the template will be used as-is
        # if GOAL > 1, the rate parameters will be updated, even in a pre-set moth

        self.TR_PER_CLASS = 1 # (try 3) the number of training samples per class
        self.NUM_SNIFFS = 1 # (try 2) number of exposures each training sample

        # nearest neighbors
        self.NUM_NEIGHBORS = 1 # optimization param for nearest neighbors
        # Suggested values:
        #	NUM_NEIGHBORS:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

        # SVM
        self.BOX_CONSTRAINT = 1e1 # optimization parameter for svm
        # Suggested values:
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

        self.RESULTS_FOLDER = _os.path.dirname(__file__) + _os.sep + 'results' # string
        # (relative path, starting inside the directory housing this package)
        # If non-empty, results will be saved here
        self.RESULTS_FILENAME = 'results' # will get the run number appended to it

        #-------------------------------------------------------------------------------

        # Test parameters for compatibility
        if self.SHOW_ACC_PLOTS or self.SHOW_TIME_PLOTS:
        	##TEST that directory string is not empty
        	if not self.RESULTS_FOLDER:
        		folder_error = "RESULTS_FOLDER parameter is empty.\n" + \
        			"Please add directory or set SHOW_ACC_PLOTS and SHOW_TIME_PLOTS to 'False'."
        		raise Exception(folder_error)

        	##TEST for existence of image results folder, else create it
        	if not _os.path.isdir(self.RESULTS_FOLDER):
        		_os.mkdir(self.RESULTS_FOLDER)
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

        self._class_labels = _np.array(range(10))  # For MNIST. '0' is labeled as 10
        self._val_per_class = 15  # number of digits used in validation sets and in baseline sets

        # make a vector of the classes of the training samples, randomly mixed:
        self._tr_classes = _np.repeat( self._class_labels, self.TR_PER_CLASS )
        self._tr_classes = _np.random.permutation( self._tr_classes )
        # repeat these inputs if taking multiple sniffs of each training sample:
        self._tr_classes = _np.tile( self._tr_classes, [1, self.NUM_SNIFFS] )[0]

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
        digit_queues = _np.zeros_like(self._feat_array)

        for i in self._class_labels:

            ## 1. Baseline (pre-train) images
            # choose some images from the baselineIndPool
            _range_top_end = max(self._ind_pool_baseline) - min(self._ind_pool_baseline) + 1
            _r_sample = _np.random.choice(_range_top_end, self._val_per_class) # select random digits
            _these_inds = min(self._ind_pool_baseline) + _r_sample
            digit_queues[:,:self._val_per_class,i] = self._feat_array[:,_these_inds,i]

            ## 2. Training images
            # choose some images from the trainingIndPool
            _range_top_end = max(self._ind_pool_train) - min(self._ind_pool_train) + 1
            _r_sample = _np.random.choice(_range_top_end, self.TR_PER_CLASS) # select random digits
            _these_inds = min(self._ind_pool_train) + _r_sample
            # repeat these inputs if taking multiple sniffs of each training sample
            _these_inds = _np.tile(_these_inds, self.NUM_SNIFFS)
            digit_queues[:, self._val_per_class:(self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS), i] = \
                self._feat_array[:, _these_inds, i]

            ## 3. Post-training (val) images
            # choose some images from the postTrainIndPool
            _range_top_end = max(self._ind_pool_post) - min(self._ind_pool_post) + 1
            _r_sample = _np.random.choice(_range_top_end, self._val_per_class) # select random digits
            _these_inds = min(self._ind_pool_post) + _r_sample
            digit_queues[:,(self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS): \
                (self._val_per_class+self.TR_PER_CLASS*self.NUM_SNIFFS+self._val_per_class),
    			i] = self._feat_array[:, _these_inds, i]

        # show the final versions of thumbnails to be used, if wished
        if self.N_THUMBNAILS:
            from modules.show_figs import show_FA_thumbs
            _thumb_array = _np.zeros((self._len_side, self._num_per_class, self._class_num))
            _thumb_array[self._active_pixel_inds,:,:] = digit_queues
            normalize = 1
            show_FA_thumbs(_thumb_array, self.N_THUMBNAILS, normalize, 'Input thumbnails',
    			self.SCREEN_SIZE, self.RESULTS_FOLDER + _os.sep + 'thumbnails')

        return digit_queues

#-------------------------------------------------------------------------------

    def train_test_split(self, digit_queues):
        '''
        Subsample the dataset for this simulation,
        then build train and val feature matrices and class label vectors.

        Returns: train_X, test_X, train_y, test_y
        '''

        # X = n x numberPixels;  Y = n x 1, where n = 10*TR_PER_CLASS.
        train_X = _np.zeros((10*self.TR_PER_CLASS, self._feat_array.shape[0]))
        test_X = _np.zeros((10*self._val_per_class, self._feat_array.shape[0]))
        train_y = _np.zeros((10*self.TR_PER_CLASS, 1))
        test_y = _np.zeros((10*self._val_per_class, 1))

        # populate the labels one class at a time
        for i in self._class_labels:
            # skip the first '_val_per_class' digits,
            # as these are used as baseline digits in the moth (formality)
            temp = digit_queues[:,self._val_per_class:self._val_per_class+self.TR_PER_CLASS,i]
            train_X[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS,:] = temp.T
            temp = digit_queues[:,self._val_per_class+self.TR_PER_CLASS: \
                2*self._val_per_class+self.TR_PER_CLASS,i]
            test_X[i*self._val_per_class:(i+1)*self._val_per_class,:] = temp.T
            train_y[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS] = i
            test_y[i*self._val_per_class:(i+1)*self._val_per_class,:] = i

        return train_X, test_X, train_y, test_y

    def load_moth(self):
        '''
        Create a new moth, ie the template that is used to populate connection
            matrices and to control behavior.
        Returns: self.model_params (Model parameters)
        '''
        from modules.params import ModelParams as ModelParams

    	# instantiate template params
        self.model_params = ModelParams( len(self._active_pixel_inds), self.GOAL )

    	# populate the moth's connection matrices using the model_params
        self.model_params.init_connection_matrix()

    def load_exp(self):
        '''
        Load experiment parameters, including book-keeping for time-stepped
    	   evolutions, eg when octopamine occurs, time regions to poll for digit
           responses, windowing of firing rates, etc.
        Returns: Experiment parameters
        '''

        from modules.params import ExpParams

        self.experiment_params =  ExpParams( self._tr_classes, self._class_labels, self._val_per_class )

    def simulate(self, digit_queues):
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
        return sde_wrap(self.model_params, self.experiment_params, digit_queues )

    def score_moth_on_MNIST(self, EN_resp_trained):
        '''
        Calculate the classification accuracy of MothNet on MNIST dataset.
        Prints:
            1. output_naive_log_loss: Baseline accuracy using log-likelihoods over all ENs
            2. output_trained_log_loss: Post-training accuracy using log-likelihoods over all ENs
            3. output_naive_thresholding: Baseline accuracy using single EN thresholding
            4. output_trained_thresholding: Post-training accuracy using single EN thresholding
        Returns:
            1. output_trained_thresholding: Post-training accuracy using single EN thresholding
            2. output_trained_log_loss: Post-training accuracy using log-likelihoods over all ENs

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''
        from modules.classify import classify_digits_log_likelihood, classify_digits_thresholding

        # for baseline accuracy function argin, substitute pre- for post-values in EN_resp_trained:
        EN_resp_naive = _copy.deepcopy(EN_resp_trained)
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
    		'by class: {}%'.format(_np.round(output_naive_log_loss['acc_perc'])))
        print(' Trained Accuracy: {}%,'.format(round(output_trained_log_loss['total_acc'])) + \
    		'by class: {}%'.format(_np.round(output_trained_log_loss['acc_perc'])))

        # 2. using single EN thresholding:
        output_naive_thresholding = classify_digits_thresholding( EN_resp_naive, 1e9, -1, 10 )
        output_trained_thresholding = classify_digits_thresholding( EN_resp_trained, 1e9, -1, 10 )

        print('Thresholding:')
        print(' Baseline (Naive) Accuracy: {}%,'.format(round(output_naive_thresholding['total_acc'])) + \
    		'by class: {}%'.format(_np.round(output_naive_thresholding['acc_perc'])))
        print(' Trained Accuracy: {}%,'.format(round(output_trained_thresholding['total_acc'])) + \
    		'by class: {}%'.format(_np.round(output_trained_thresholding['acc_perc'])))

        if self.SHOW_ROC_PLOTS:
            from modules.show_figs import show_roc_curves
            # compute macro-average ROC curve
            show_roc_curves(output_trained_log_loss['fpr'], output_trained_log_loss['tpr'],
        		output_trained_log_loss['roc_auc'], self._class_labels,
        		title_str='MothNet',
                images_filename=self.RESULTS_FOLDER + _os.sep + self.RESULTS_FILENAME)

        self.output_trained_thresholding = output_trained_thresholding
        self.output_trained_log_loss = output_trained_log_loss

    def score_knn(self, train_X, train_y, test_X, test_y):
        '''
        Calculate the classification accuracy of KNN on MNIST dataset.
        Parameters:
            1. train_X: Feature matrix training samples
            2. train_y: Labels for training samples
            3. test_X: Feature matrix testing samples
            4. test_y: Labels for testing samples
        Prints:
            1. output_naive_log_loss: Baseline accuracy using log-likelihoods over all ENs
            2. output_trained_log_loss: Post-training accuracy using log-likelihoods over all ENs
            3. output_naive_thresholding: Baseline accuracy using single EN thresholding
            4. output_trained_thresholding: Baseline accuracy using single EN thresholding
        Returns:
            self.output_trained_log_loss

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''

        ##TEST to see if sklearn is installed,
        try:
            from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
        except ImportError:
            print('sklearn is not installed, and it is required to run KNN models.\n')

        neigh = _KNeighborsClassifier(n_neighbors=self.NUM_NEIGHBORS)
        neigh.fit(train_X, train_y.ravel())
        y_hat = neigh.predict(test_X)

        # get probabilities
        probabilities = neigh.predict_proba(test_X)

        if self.SHOW_ROC_PLOTS:
            from modules.classify import roc_multi
            from modules.show_figs import show_roc_curves

            # measure ROC AUC for each class
            self.roc_knn = roc_multi(test_y.flatten(), probabilities)

            # compute macro-average ROC curve
            show_roc_curves(self.roc_knn['fpr'], self.roc_knn['tpr'],
                self.roc_knn['roc_auc'], self._class_labels, title_str='KNN',
				images_filename=self.RESULTS_FOLDER + _os.sep + self.RESULTS_FILENAME)

        # measure overall accuracy
        nn_acc = neigh.score(test_X, test_y)

        # measure accuracy for each class
        class_acc = _np.zeros_like(self._class_labels) # preallocate
        for i in self._class_labels:
            inds = _np.where(test_y==i)[0]
            class_acc[i] = _np.round(100*_np.sum( y_hat[inds]==test_y[inds].squeeze()) /
                len(test_y[inds]) )

        print('Nearest neighbor (k[# of neighbors]={}):\n'.format(self.NUM_NEIGHBORS),
            'Trained Accuracy = {}%,'.format(_np.round(100*nn_acc)),
            'by class: {}% '.format(class_acc) )

    def score_svm(self, train_X, train_y, test_X, test_y):
        '''
        Calculate the classification accuracy of SVM on MNIST dataset.
        Parameters:
            1. train_X: Feature matrix training samples
            2. train_y: Labels for training samples
            3. test_X: Feature matrix testing samples
            4. test_y: Labels for testing samples
        Prints:
            1. output_naive_log_loss: Baseline accuracy using log-likelihoods over all ENs
            2. output_trained_log_loss: Post-training accuracy using log-likelihoods over all ENs
            3. output_naive_thresholding: Baseline accuracy using single EN thresholding
            4. output_trained_thresholding: Baseline accuracy using single EN thresholding
        Returns:
            self.output_trained_log_loss

        Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
        MIT License
        '''

        ##TEST to see if sklearn is installed
        try:
            from sklearn import svm as _svm
        except ImportError:
            print('sklearn is not installed, and it is required to run KNN models.\n')

        svm_clf = _svm.SVC(gamma='scale', C=self.BOX_CONSTRAINT, probability=True)
        svm_clf.fit(train_X, train_y.ravel())
        y_hat = svm_clf.predict(test_X)

		# get probabilities
        probabilities = svm_clf.predict_proba(test_X)

        if self.SHOW_ROC_PLOTS:
            from modules.classify import roc_multi
            from modules.show_figs import show_roc_curves

            # measure ROC AUC for each class
            self.roc_svm = roc_multi(test_y.flatten(), probabilities)

            # compute macro-average ROC curve
            show_roc_curves(self.roc_svm['fpr'], self.roc_svm['tpr'],
                self.roc_svm['roc_auc'], self._class_labels, title_str='KNN',
				images_filename=self.RESULTS_FOLDER + _os.sep + self.RESULTS_FILENAME)

        # measure overall accuracy
        svm_acc = svm_clf.score(test_X, test_y)

        # measure accuracy for each class
        class_acc = _np.zeros_like(self._class_labels) # preallocate
        for i in self._class_labels:
            inds = _np.where(test_y==i)[0]
            class_acc[i] = _np.round(100*_np.sum(y_hat[inds]==test_y[inds].squeeze()) /
                len(test_y[inds]))

        print('Nearest neighbor (k[# of neighbors]={}):\n'.format(self.NUM_NEIGHBORS),
            'Trained Accuracy = {}%,'.format(_np.round(100*svm_acc)),
            'by class: {}% '.format(class_acc))

    def show_multi_roc(self, model_names, class_labels, images_filename=''):
        '''
            show_multi_roc(model_names, class_labels, images_filename)
        '''
        import matplotlib.pyplot as plt
        from modules.show_figs import plot_roc_multi

        if images_filename:
            images_folder = _os.path.dirname(images_filename)

            # create directory for images (if doesnt exist)
            if images_folder and not _os.path.isdir(images_folder):
                _os.mkdir(images_folder)
                print('Creating results directory: {}'.format(images_folder))

        roc_dict_list = [self.output_trained_log_loss, self.roc_svm, self.roc_knn]

        fig, axes = plt.subplots(1, len(roc_dict_list), figsize=(15,5), sharey=True)

        y_ax_list = [True, False, False]
        legend_list = [True, False, False]

        for i in range(len(roc_dict_list)):
            ax = axes[i]
            fpr = roc_dict_list[i]['fpr']
            tpr = roc_dict_list[i]['tpr']
            roc_auc = roc_dict_list[i]['roc_auc']
            title_str = model_names[i]

            plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str,
                y_axis_label=y_ax_list[i], legend=legend_list[i])

        fig.tight_layout()

        # save plot
        if _os.path.isdir(images_folder):
            roc_filename = images_filename + '.png'
            fig.savefig(roc_filename, dpi=150)
            print(f'Figure saved: {roc_filename}')
        else:
            print('ROC curves NOT SAVED!\nMake sure a valid directory path has been prepended to `images_filename`')
