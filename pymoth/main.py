#!/usr/bin/env python3

"""

.. module:: pymoth
   :platform: Unix
   :synopsis: The core module of pymoth package.

.. moduleauthor:: Adam P. Jones <ajones173@gmail.com>

"""

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

	# Experiment details
	from .modules.sde import collect_stats
	from .modules.show_figs import show_multi_roc

	# Initializer / Instance Attributes
	def __init__(self, settings):
		"""

		Python class to train a moth brain model on a crude (downsampled) MNIST set. \
		The moth can be generated from template or loaded complete from file.

		Because the moth brain architecture, as evolved, only handles ~60 features, \
		we need to create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features. \
		We do this by cropping and downsampling the MNIST thumbnails, then selecting \
		a subset of the remaining pixels.

		This results in a cruder dataset (set various view flags to see thumbnails). \
		However, it is sufficient for testing the moth brain's learning ability. Other ML \
		methods need to be tested on this same cruder dataset to make useful comparisons.

		Defining train and control pools for the experiment determines the receptive \
		field. This is done first because the receptive field determines the number \
		of AL units, which must be updated in model_params before :func:`create_connection_matrix` \
		runs.

		This dataset will be used for each simulation in numRuns. Each simulation draws \
		a new set of samples from this set.

		Order of events:

		#. Load and pre-process dataset
		#. During simulation, select a subset of the dataset for this simulation (only a few samples are used)
		#.  Create a moth (neural net). Either select an existing moth file, or generate \
		a new moth in 2 steps:
			* First, run :class:`ModelParams` and incorporate user entry edits such as GOAL
			* Second, create connection matrices via :func:`create_connection_matrix`

		#. Load the experiment parameters
		#. Run the simulation with :func:`sde_wrap`, print results to console
		#. Plot results (optional)
		#. Run addition ML models for comparison, print results to console (optional)

		Args:
			user_params (dict):  Package configuration values:
				SCREEN_SIZE (tuple): screen size (width, height)
				NUM_RUNS (int): how many runs you wish to do with this moth. Each run \
				using random draws from the MNIST set.
				GOAL (int): the moth's learning rates, in terms of how many training \
				samples per class give max accuracy. So '1' gives a very fast learner.
				TR_PER_CLASS (int): number of training samples per class (try 3)
				NUM_SNIFFS (int): number of exposures for each training sample (try 2)
				NUM_NEIGHBORS (int): hyper-param for nearest neighbors (try 1)
				BOX_CONSTRAINT (float): optimization parameter for SVM (try 1e1)
				N_THUMBNAILS (int): flag to show N experiment inputs from each class \
				(0 means don't show any).
				SHOW_ACC_PLOTS (bool): True to plot, False to ignore. One image \
				(with 8 subplots) per EN.
				SHOW_TIME_PLOTS (bool): True to plot, False to ignore. Three scaled \
				ENs timecourses on each of 4 images (only one EN on the 4th image).
				SHOW_ROC_PLOTS (bool): True to plot, False to ignore. One for each model.
				RESULTS_FOLDER (str): relative path, starting from user's home directory. \
				If non-empty, results will be saved here.
				RESULTS_FILENAME (str): will get the run number appended to it.
				DATA_FOLDER (str): relative path, starting from user's home directory.
				DATA_FILENAME (str): filename for MNIST data.

		>>> mothra = pymoth.MothNet()

		"""

		# package config values (constants)
		self.SCREEN_SIZE = settings.get('screen_size', (1920, 1080)) # screen size (width, height)
		self.NUM_RUNS = settings.get('num_runs', 1) # how many runs you wish to do with this moth
		self.GOAL = settings.get('goal', 15) # define the moth's learning rates
		self.TR_PER_CLASS = settings.get('tr_per_class', 1) # number of training samples per class
		self.NUM_SNIFFS = settings.get('num_sniffs', 1) # number of exposures each training sample
		self.NUM_NEIGHBORS = settings.get('num_neighbors', 1) # optimization param for nearest neighbors
		self.BOX_CONSTRAINT = settings.get('box_constraint', 1e1) # optimization parameter for svm
		self.N_THUMBNAILS = settings.get('n_thumbnails', 1) # show N experiment inputs from each class
		self.SHOW_ACC_PLOTS = settings.get('show_acc_plots', True) # True to plot, False to ignore
		self.SHOW_TIME_PLOTS = settings.get('show_time_plots', True) # True to plot, False to ignore
		self.SHOW_ROC_PLOTS = settings.get('show_roc_plots', True) # True to plot, False to ignore
		self.RESULTS_FOLDER = settings.get('results_folder', '/tmp') # string
		self.RESULTS_FILENAME = settings.get('results_filename', 'results') # string
		self.DATA_FOLDER = settings.get('data_folder', '/tmp') # string
		self.DATA_FILENAME = settings.get('data_filename', 'MNIST_all') # string

		# Test parameters for compatibility
		if self.SHOW_ACC_PLOTS or self.SHOW_TIME_PLOTS:
			##TEST that directory string is not empty
			if self.RESULTS_FOLDER!='/tmp':
				self.RESULTS_FOLDER = _os.path.expanduser("~")+_os.sep+self.RESULTS_FOLDER

			##TEST for existence of image results folder, else create it
			if not _os.path.isdir(self.RESULTS_FOLDER):
				_os.mkdir(self.RESULTS_FOLDER)
				print('\nCreating results directory: {}\n'.format(self.RESULTS_FOLDER))

	### 2. Load and preprocess MNIST dataset ###

	def load_mnist(self):
		"""
		Load and preprocess MNIST dataset

		Because the moth brain architecture, as evolved, only handles ~60 features, \
		we need to create a new, MNIST-like task but with many fewer than 28x28 \
		pixels-as-features. We do this by cropping and downsampling the MNIST thumbnails, \
		then selecting a subset of the remaining pixels.

		This results in a cruder dataset (set various view flags to see thumbnails). \
		However, it is sufficient for testing the moth brain's learning ability. \
		Other ML methods need to be tested on this same cruder dataset to make \
		useful comparisons.

		Define train and control pools for the experiment, and determine the receptive \
		field. This is done first because the receptive field determines the number \
		of AL units. This dataset will be used for each simulation in NUM_RUNS. \
		Each simulation draws a new set of samples from this set.

		Args:
			None

		Returns
		-------
			feature_array (numpy array): stimuli [numFeatures x numStimsPerClass x numClasses]

		>>> mothra.load_mnist()
		"""

		from .modules.generate import generate_ds_mnist

		self._class_labels = _np.array(range(10)) # MNIST classes: digits 0-9
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
		self._inds_to_calc_RF = list(range(550,1000)) # pixel indices for receptive field
		self._max_ind = max( [ self._inds_to_calc_RF + self._ind_pool_train ][0] ) # we'll throw out unused samples

		## 2. Pre-processing parameters for the thumbnails:
		self._downsample_rate = 2 # image downsampling ratio (n:1)
		self._crop = 2 # image cropping parameter
		self._num_features = 85 # number of pixels in the receptive field
		self._pixel_sum = 6 # normalization factor
		self._show_thumbnails = self.N_THUMBNAILS
		self._downsample_method = 1 # 0 means sum square patches of pixels
							# 1 means use bicubic interpolation

		# generate the data array:
		# _feat_array is a feature array ready for running experiments.
		# Each experiment uses a random draw from this dataset.
		self._feat_array, self._active_pixel_inds, self._len_side = generate_ds_mnist(
			self._max_ind, self._class_labels, self._crop, self._downsample_rate,
			self._downsample_method, self._inds_to_ave, self._pixel_sum,
			self._inds_to_calc_RF, self._num_features, self.SCREEN_SIZE,
			self.RESULTS_FOLDER, self._show_thumbnails,
			data_dir = self.DATA_FOLDER, data_fname = self.DATA_FILENAME
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
			from .modules.show_figs import show_FA_thumbs
			_thumb_array = _np.zeros((self._len_side, self._num_per_class, self._class_num))
			_thumb_array[self._active_pixel_inds,:,:] = digit_queues
			normalize = 1
			show_FA_thumbs(_thumb_array, self.N_THUMBNAILS, normalize, 'Input thumbnails',
				self.SCREEN_SIZE, self.RESULTS_FOLDER + _os.sep + 'thumbnails')

		return digit_queues

#-------------------------------------------------------------------------------

	def train_test_split(self, feature_array):
		"""

		Subsample the dataset for this simulation, then build train and val feature \
		matrices and class label vectors.

		Args:
			feature_array (numpy array): Stimuli (numFeatures x numStimsPerClass x numClasses)

		Returns
		-------
			train_X (numpy array)
				Feature matrix training samples
			test_X (numpy array)
				Feature matrix testing samples
			train_y (numpy array)
				Labels for training samples
			test_y (numpy array)
				Labels for testing samples

		>>> train_X, test_X, train_y, test_y = mothra.train_test_split(feature_array)

		"""

		# X = n x numberPixels;  Y = n x 1, where n = 10*TR_PER_CLASS.
		train_X = _np.zeros((10*self.TR_PER_CLASS, self._feat_array.shape[0]))
		test_X = _np.zeros((10*self._val_per_class, self._feat_array.shape[0]))
		train_y = _np.zeros((10*self.TR_PER_CLASS, 1))
		test_y = _np.zeros((10*self._val_per_class, 1))

		# populate the labels one class at a time
		for i in self._class_labels:
			# skip the first '_val_per_class' digits,
			# as these are used as baseline digits in the moth (formality)
			temp = feature_array[:,self._val_per_class:self._val_per_class+self.TR_PER_CLASS,i]
			train_X[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS,:] = temp.T
			temp = feature_array[:,self._val_per_class+self.TR_PER_CLASS: \
				2*self._val_per_class+self.TR_PER_CLASS,i]
			test_X[i*self._val_per_class:(i+1)*self._val_per_class,:] = temp.T
			train_y[i*self.TR_PER_CLASS:(i+1)*self.TR_PER_CLASS] = i
			test_y[i*self._val_per_class:(i+1)*self._val_per_class,:] = i

		return train_X, test_X, train_y, test_y

	def load_moth(self):
		"""

		Create a new moth, ie the template that is used to populate connection \
		matrices and to control behavior.

		Args:
			None

		Returns
		-------
			model_params (class)
				Model parameters.

		>>> mothra.load_moth()

		"""
		from .modules.params import ModelParams

		# instantiate template params
		self.model_params = ModelParams( len(self._active_pixel_inds), self.GOAL )

		# populate the moth's connection matrices using the model_params
		self.model_params.create_connection_matrix()

	def load_exp(self):
		"""

		Load experiment parameters, including book-keeping for time-stepped \
		evolutions, eg when octopamine occurs, time regions to poll for digit \
		responses, windowing of firing rates, etc.

		Creates self.experiment_params.

		Args:
			None

		Returns
		-------
			None

		>>> mothra.load_exp()

		"""

		from .modules.params import ExpParams
		self.experiment_params =  ExpParams( self._tr_classes, self._class_labels, self._val_per_class )

	def simulate(self, feature_array):
		"""

		Run the SDE time-stepped evolution of neural firing rates.

		Steps:
			#. Load various params needed for pre-evolution prep.
			#. Specify stim and octo courses.
			#. Interaction equations and step through simulation.
			#. Unpack evolution output and export.

		Args:
			feature_array (numpy array): array of stimuli [num_features X \
			num_stims_per_class X num_classes]

		Returns
		-------
			sim_results (dict)
				EN timecourses and final P2K and K2E connection matrices.

		>>> sim_results = mothra.simulate(feature_array)

		"""
		from .modules.sde import sde_wrap

		print('\nStarting sim for goal = {}, tr_per_class = {}, numSniffsPerSample = {}'.format(
			self.GOAL, self.TR_PER_CLASS, self.NUM_SNIFFS))

		# run this experiment as sde time-step evolution:
		return sde_wrap(self.model_params, self.experiment_params, feature_array )

	def score_moth_on_MNIST(self, EN_resp_trained):
		"""

		Calculate the classification accuracy of MothNet on MNIST dataset.

		Sets:
			self.output_trained_thresholding (dict): Post-training accuracy using \
			single EN thresholding.
			self.output_trained_log_loss (dict): Post-training accuracy using \
			log-likelihoods over all ENs.

		Args:
			EN_resp_trained (list): simulation EN responses grouped by class and time.

		Returns
		-------
			None

		>>> mothra.score_moth_on_MNIST(EN_resp_trained)

		"""
		from .modules.classify import classify_digits_log_likelihood, classify_digits_thresholding

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
			from .modules.show_figs import show_roc_curves
			# compute macro-average ROC curve
			show_roc_curves(output_trained_log_loss['tpr'], output_trained_log_loss['fpr'],
				output_trained_log_loss['roc_auc'], self._class_labels,
				title_str='MothNet',
				images_filename=self.RESULTS_FOLDER + _os.sep + self.RESULTS_FILENAME)

		self.output_trained_thresholding = output_trained_thresholding
		self.output_trained_log_loss = output_trained_log_loss

	def score_knn(self, train_X, train_y, test_X, test_y):
		"""

		Calculate the classification accuracy of KNN on MNIST dataset and print \
		the accuracy.

		Args:
			train_X (numpy array): Feature matrix training samples
			train_y (numpy array): Labels for training samples
			test_X (numpy array): Feature matrix testing samples
			test_y (numpy array): Labels for testing samples

		Returns
		-------
			None

		>>> mothra.score_knn(train_X, train_y, test_X, test_y)

		"""

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
			from .modules.classify import roc_multi
			from .modules.show_figs import show_roc_curves

			# measure ROC AUC for each class
			self.roc_knn = roc_multi(test_y.flatten(), probabilities)

			# compute macro-average ROC curve
			show_roc_curves(self.roc_knn['tpr'], self.roc_knn['fpr'],
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
		"""

		Calculate the classification accuracy of SVM on MNIST dataset and print \
		the accuracy.

		Args:
			train_X (numpy array): Feature matrix training samples
			train_y (numpy array): Labels for training samples
			test_X (numpy array): Feature matrix testing samples
			test_y (numpy array): Labels for testing samples

		Returns
		-------
			None

		>>> mothra.score_svm(train_X, train_y, test_X, test_y)

		"""

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
			from .modules.classify import roc_multi
			from .modules.show_figs import show_roc_curves

			# measure ROC AUC for each class
			self.roc_svm = roc_multi(test_y.flatten(), probabilities)

			# compute macro-average ROC curve
			show_roc_curves(self.roc_svm['tpr'], self.roc_svm['fpr'],
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
