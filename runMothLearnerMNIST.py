'''
runMothLearnerOnReducedMnist

Main script to train a moth brain model on a crude (downsampled) MNIST set.
The moth can be generated from template or loaded complete from file.

Preparation:
	1.  Modify 'specifyModelParamsMnist_fn' with the desired parameters for
		generating a moth (ie neural network), or specify a pre-existing 'modelParams' file to load.
	2. Edit USER ENTRIES

Order of events:
	1. Load and pre-process dataset
	Within the loop over number of simulations:
	2. Select a subset of the dataset for this simulation (only a few samples are used).
	3. Create a moth (neural net). Either select an existing moth file, or generate a new moth in 2 steps:
		a) run 'specifyModelParamsMnist' and
		   incorporate user entry edits such as 'goal'.
		b) create connection matrices via 'initializeConnectionMatrices'
	4. Load the experiment parameters.
	5. Run the simulation with 'sdeWrapper'
	6. Plot results, print results to console

Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
MIT License
'''

# import packages
import time
runStart = time.time() # time execution duration
import numpy as np
import os
import dill # for pickling module object (optional)
import copy # for deep copy of nested lists

# Experiment details
from support_functions.genDS_MNIST import generateDownsampledMNISTSet
from support_functions.show_figs import showFeatureArrayThumbnails, viewENresponses
from support_functions.connect_mat import initializeConnectionMatrices
from support_functions.setMNISTExpParams import setMNISTExperimentParams
from support_functions.sdeWrap import sdeWrapper
from support_functions.classifyDigits import classifyDigitsViaLogLikelihood, classifyDigitsViaThresholding

## USER ENTRIES (Edit parameters below):
#-------------------------------------------------------------------------------
scrsz = (1920, 1080) # screen size (width, height)

useExistingConnectionMatrices = False
# if True, load 'matrixParamsFilename', which includes filled-in connection matrices
# if False, generate new moth from template in specifyModelParamsMnist_fn.py

matrixParamsFilename = 'sampleMothModelParams'
# dict with all info, including connection matrices, of a particular moth

numRuns = 1 # how many runs you wish to do with this moth or moth template,
# each run using random draws from the mnist set

goal  = 15
# defines the moth's learning rates, in terms of how many training samples per
# class give max accuracy. So "goal = 1" gives a very fast learner.
# if goal == 0, the rate parameters defined the template will be used as-is
# if goal > 1, the rate parameters will be updated, even in a pre-set moth

trPerClass =  3 # the number of training samples per class
numSniffs = 2 # number of exposures each training sample

## Flags to show various images:
showThumbnails = 0 # N means show N experiment inputs from each class
	# 0 means don't show any
showENPlots = [1, 1] # 1 to plot, 0 to ignore
# arg1 refers to statistical plots of EN response changes: One image (with 8 subplots) per EN
# arg2 refers to EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image)

# To save results if wished:
saveAllNeuralTimecourses = False # 0 -> save only EN (ie readout) timecourses
# Caution: 1 -> very high memory demands, hinders longer runs
resultsFilename = 'results' # will get the run number appended to it
saveResultsDataFolder = 'results/data' # String
# If non-empty, 'resultsFilename' will be saved here
saveResultsImageFolder = 'results' # String
# If non-empty, results will be saved here (if showENPlots also non-zero)
saveParamsFolder = 'params' # String
# If non-empty, params will be saved here (if showENPlots also non-zero)

#-------------------------------------------------------------------------------

## Misc book-keeping
classLabels = np.array(range(10))  # For MNIST. '0' is labeled as 10
valPerClass = 15  # number of digits used in validation sets and in baseline sets

# make a vector of the classes of the training samples, randomly mixed:
trClasses = np.repeat( classLabels, trPerClass )
trClasses = np.random.permutation( trClasses )
# repeat these inputs if taking multiple sniffs of each training sample:
trClasses = np.tile( trClasses, [1, numSniffs] )[0]

# Experiment details for 10 digit training:
experimentFn = setMNISTExperimentParams

#-------------------------------------------------------------------------------

## Load and preprocess the dataset.

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
#      must be updated in modelParams before 'initializeMatrixParams_fn' runs.
# This dataset will be used for each simulation in numRuns. Each
#      simulation draws a new set of samples from this set.

# Parameters:
# Parameters required for the dataset generation function are attached to a dictionary: 'preP'.
# 1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
#   This is NOT the number of training samples per class.
# 	That is trPerClass, defined above.

# Specify pools of indices from which to draw baseline, train, val sets.
indPoolForBaseline = list(range(100)) # 1:100
indPoolForTrain = list(range(100,300)) # 101:300
indPoolForPostTrain = list(range(300,400)) # 301:400

# Population preprocessing pools of indices:
preP = dict()
preP['indsToAverageGeneral'] = list(range(550,1000)) # 551:1000
preP['indsToCalculateReceptiveField'] = list(range(550,1000)) # 551:1000
preP['maxInd'] = max( [ preP['indsToCalculateReceptiveField'] + \
	indPoolForTrain ][0] ) # we'll throw out unused samples

## 2. Pre-processing parameters for the thumbnails:
preP['downsampleRate'] = 2
preP['crop'] = 2
preP['numFeatures'] =  85  # number of pixels in the receptive field
preP['pixelSum'] = 6
preP['showThumbnails'] = showThumbnails # boolean
preP['downsampleMethod'] = 1 # 0 means sum square patches of pixels
							 # 1 means use bicubic interpolation

preP['classLabels'] = classLabels # append
preP['useExistingConnectionMatrices'] = useExistingConnectionMatrices # boolean
preP['matrixParamsFilename'] = matrixParamsFilename

# generate the data array:
fA, activePixelInds, lengthOfSide = generateDownsampledMNISTSet(preP, saveResultsImageFolder, scrsz)
# argin = preprocessingParams

pixNum, numPerClass, classNum = fA.shape
# The dataset fA is a feature array ready for running experiments.
# Each experiment uses a random draw from this dataset.
# fA = n x m x 10 array where n = #active pixels, m = #digits from each class
# that will be used. The 3rd dimension gives the class, 1:10 where 10 = '0'.

#-------------------------------------------------------------------------------

# Loop through the number of simulations specified:
print(f'starting sim(s) for goal = {goal}, trPerClass = {trPerClass}, numSniffsPerSample = {numSniffs}')

for run in range(numRuns):

	## Subsample the dataset for this simulation
	# Line up the images for the experiment (in 10 parallel queues)
	digitQueues = np.zeros(fA.shape)

	for i in classLabels:

		## 1. Baseline (pre-train) images
		# choose some images from the baselineIndPool
		rangeTopEnd = max(indPoolForBaseline) - min(indPoolForBaseline) + 1
		r_sample = np.random.choice(rangeTopEnd, valPerClass) # select random digits
		theseInds = min(indPoolForBaseline) + r_sample
		digitQueues[:,:valPerClass,i] = fA[:,theseInds,i]

		## 2. Training images
		# choose some images from the trainingIndPool
		rangeTopEnd = max(indPoolForTrain) - min(indPoolForTrain) + 1
		r_sample = np.random.choice(rangeTopEnd, trPerClass) # select random digits
		theseInds = min(indPoolForTrain) + r_sample
		# repeat these inputs if taking multiple sniffs of each training sample
		theseInds = np.tile(theseInds, numSniffs)
		digitQueues[:, valPerClass:(valPerClass+trPerClass*numSniffs), i] = fA[:, theseInds, i]

		## 3. Post-training (val) images
		# choose some images from the postTrainIndPool
		rangeTopEnd = max(indPoolForPostTrain) - min(indPoolForPostTrain) + 1
		r_sample = np.random.choice(rangeTopEnd, valPerClass) # select random digits
		theseInds = min(indPoolForPostTrain) + r_sample
		digitQueues[:,(valPerClass+trPerClass*numSniffs):(valPerClass+trPerClass*numSniffs+valPerClass),
			i] = fA[:, theseInds, i]

	# show the final versions of thumbnails to be used, if wished
	if showThumbnails:
		tempArray = np.zeros((lengthOfSide, numPerClass, classNum))
		tempArray[activePixelInds,:,:] = digitQueues
		normalize = 1
		titleStr = 'Input thumbnails'
		showFeatureArrayThumbnails(tempArray, showThumbnails, normalize,
									titleStr, scrsz, saveResultsImageFolder)

#-------------------------------------------------------------------------------
	# Create a moth. Either load an existing moth, or create a new moth
	if useExistingConnectionMatrices:
		params_fname = os.path.join(saveParamsFolder, 'modelParams.pkl')
		# load modelParams
		with open(params_fname,'rb') as f:
			modelParams = dill.load(f)

	else:
		# Load template params
		from support_functions.specifyModelParamsMnist import ModelParams
		modelParams = ModelParams(nF=len(activePixelInds), goal=goal)

		# Now populate the moth's connection matrices using the modelParams
		modelParams = initializeConnectionMatrices(modelParams)

		# save params to file (if saveParamsFolder not empty)
		if saveParamsFolder:
			if not os.path.isdir(saveParamsFolder):
				os.mkdir(saveParamsFolder)
				# pickle parameters for other branch of if construct
				params_fname = os.path.join(saveParamsFolder, 'modelParams.pkl')
				dill.dump(modelParams, open(params_fname, 'wb'))

	modelParams.trueClassLabels = classLabels # misc parameter tagging along
	modelParams.saveAllNeuralTimecourses = saveAllNeuralTimecourses

	# Define the experiment parameters, including book-keeping for time-stepped
	# 	evolutions, eg when octopamine occurs, time regions to poll for digit
	# 	responses, windowing of Firing rates, etc
	experimentParams = experimentFn( trClasses, classLabels, valPerClass )

#-------------------------------------------------------------------------------

	# 3. run this experiment as sde time-step evolution:
	simResults = sdeWrapper( modelParams, experimentParams, digitQueues )

#-------------------------------------------------------------------------------

	# Experiment Results: EN behavior, classifier calculations:
	if saveResultsImageFolder:
		if not os.path.isdir(saveResultsImageFolder):
			os.mkdir(saveResultsImageFolder)

	# Process the sim results to group EN responses by class and time:
	respOrig = viewENresponses(simResults, modelParams, experimentParams,
		showENPlots, classLabels, scrsz, resultsFilename, saveResultsImageFolder)

	# Calculate the classification accuracy:
	# for baseline accuracy function argin, substitute pre- for post-values in respOrig:
	respNaive = copy.deepcopy(respOrig)
	for i, resp in enumerate(respOrig):
		respNaive[i]['postMeanResp'] = resp['preMeanResp'].copy()
		respNaive[i]['postStdResp'] = resp['preStdResp'].copy()
		respNaive[i]['postTrainOdorResp'] = resp['preTrainOdorResp'].copy()

	# 1. Using Log-likelihoods over all ENs:
	# Baseline accuracy:
	outputNaiveLogL = classifyDigitsViaLogLikelihood( respNaive )
	print( 'LogLikelihood:' )
	print( f"Naive Accuracy: {round(outputNaiveLogL['totalAccuracy'])}" + \
		f"#, by class: {np.round(outputNaiveLogL['accuracyPercentages'])} #.   ")

	# Post-training accuracy using log-likelihood over all ENs:
	outputTrainedLogL = classifyDigitsViaLogLikelihood( respOrig )
	print( f"Trained Accuracy: {round(outputTrainedLogL['totalAccuracy'])}" + \
		f"#, by class: {np.round(outputTrainedLogL['accuracyPercentages'])} #.   ")

	# 2. Using single EN thresholding:
	outputNaiveThresholding = classifyDigitsViaThresholding( respNaive, 1e9, -1, 10 )
	outputTrainedThresholding = classifyDigitsViaThresholding( respOrig, 1e9, -1, 10 )
	#     disp( 'Thresholding: ')
	#     disp( [ 'Naive accuracy: ' num2str(round(outputNaiveThresholding.totalAccuracy)),...
	#               '#, by class: ' num2str(round(outputNaiveThresholding.accuracyPercentages)), ' #.   ' ])
	#     disp([ ' Trained accuracy: ' num2str(round(outputTrainedThresholding.totalAccuracy)),...
	#               '#, by class: ' num2str(round(outputTrainedThresholding.accuracyPercentages)), ' #.   ' ])

	# append the accuracy results, and other run data, to the first entry of respOrig:
	respOrig[0]['modelParams'] = modelParams  # will include all connection weights of this moth
	respOrig[0]['outputNaiveLogL'] = outputNaiveLogL
	respOrig[0]['outputTrainedLogL'] = outputTrainedLogL
	respOrig[0]['outputNaiveThresholding'] = outputNaiveThresholding
	respOrig[0]['outputTrainedThresholding'] = outputTrainedThresholding
	respOrig[0]['matrixParamsFilename'] = matrixParamsFilename
	respOrig[0]['K2Efinal'] = simResults['K2Efinal']

	if saveResultsDataFolder:
		if not os.path.isdir(saveResultsDataFolder):
			os.mkdir(saveResultsDataFolder)

		# save results data
		results_fname = os.path.join(saveResultsDataFolder, f'{resultsFilename}_{run}.pkl')
		dill.dump(respOrig, open(results_fname, 'wb'))
		# open via:
		# >>> with open(results_fname,'rb') as f:
    	# >>> 	B = dill.load(f)

		print(f'Results saved to: {results_fname}')

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
