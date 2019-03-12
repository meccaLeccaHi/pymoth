
"""
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
		a) run 'specifyModelParamsMnist_fn' and
		   incorporate user entry edits such as 'goal'.
		b) create connection matrices via 'initializeConnectionMatrices_fn'
	4. Load the experiment parameters.
	5. Run the simulation with 'sdeWrapper_fn'
	6. Plot results, print results to console
"""

# import packages
import numpy as np

# Experiment details
from support_functions.setMNISTExpParams import setMNISTExperimentParams
from support_functions.genDS_MNIST import generateDownsampledMNISTSet
from MNIST_all.MNIST_read import MNIST_read
from support_functions.show_figs import showFeatureArrayThumbnails
import support_functions.specifyModelParamsMnist as model_params
from support_functions.connect_mat import initializeConnectionMatrices


## USER ENTRIES (Edit parameters below):

useExistingConnectionMatrices = False
# if True, load 'matrixParamsFilename', which includes filled-in connection matrices
# if False, generate new moth from template in specifyModelParamsMnist_fn.py

matrixParamsFilename = 'sampleMothModelParams' # dict with all info, including connection matrices, of a particular moth.

numRuns = 1 # how many runs you wish to do with this moth or moth template, each run using random draws from the mnist set.

goal  = 15
# defines the moth's learning rates, in terms of how many training samples per class give max accuracy. So "goal = 1" gives a very fast learner.
# if goal == 0, the rate parameters defined the template will be used as-is. if goal > 1, the rate parameters will be updated, even in a pre-set moth.

trPerClass =  3 # the number of training samples per class
numSniffs = 2 # number of exposures each training sample

## Flags to show various images:
showAverageImages = False # to show thumbnails in 'examineClassAveragesAndCorrelations_fn'
showThumbnailsUsed =  0 #  N means show N experiment inputs from each class. 0 means don't show any.
showENPlots = [1, 1] # 1 to plot, 0 to ignore
# arg1 refers to statistical plots of EN response changes: One image (with 8 subplots) per EN.
# arg2 refers to EN timecourses: Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image).

# To save results if wished:
saveAllNeuralTimecourses = False # 0 -> save only EN (ie readout) timecourses.  Caution: 1 -> very high memory demands, hinders longer runs.
resultsFilename = 'results'  # will get the run number appended to it.
saveResultsDataFolder = [] # String. If non-empty, 'resultsFilename' will be saved here.
saveResultsImageFolder = [] # StrtempArraying. If non-empty, images will be saved here (if showENPlots also non-zero).

#-----------------------------------------------

## Misc book-keeping

classLabels = list(range(10))  # For MNIST. '0' is labeled as 10
valPerClass = 15  # number of digits used in validation sets and in baseline sets

# make a vector of the classes of the training samples, randomly mixed:
trClasses = np.repeat( classLabels, trPerClass )
trClasses = np.random.permutation( trClasses )
# repeat these inputs if taking multiple sniffs of each training sample:
trClasses = np.tile( trClasses, [1, numSniffs] )

# Experiment details for 10 digit training:
experimentFn = setMNISTExperimentParams # @setMnistExperimentParams_fn

#-----------------------------------------------

## Load and preprocess the dataset.

# The dataset:
# Because the moth brain architecture, as evolved, only handles ~60 features, we need to
# create a new, MNIST-like task but with many fewer than 28x28 pixels-as-features.
# We do this by cropping and downsampling the MNIST thumbnails, then selecting a subset of the
# remaining pixels.
# This results in a cruder dataset (set various view flags to see thumbnails).
# However, it is sufficient for testing the moth brain's learning ability. Other ML methods need
# to be tested on this same cruder dataset to make useful comparisons.

# Define train and control pools for the experiment, and determine the receptive field.
# This is done first because the receptive field determines the number of AL units, which
#      must be updated in modelParams before 'initializeMatrixParams_fn' runs.experimentFn
# This dataset will be used for each simulation in numRuns. Each
#      simulation draws a new set of samples from this set.

# Parameters:
# Parameters required for the dataset generation function are attached to a dictionary: 'preP'.
# 1. The images used. This includes pools for mean-subtraction, baseline, train, and val.
#     This is NOT the number of training samples per class. That is trPerClass, defined above.

# Specify pools of indices from which to draw baseline, train, val sets.
indPoolForBaseline = list(range(100)) # 1:100
indPoolForTrain = list(range(100,300)) # 101:300
indPoolForPostTrain = list(range(300,400)) # 301:400

# Population preprocessing pools of indices:
preP = dict()
preP['indsToAverageGeneral'] = list(range(550,999)) # 551:1000
preP['indsToCalculateReceptiveField'] = list(range(550,999)) # 551:1000
preP['maxInd'] = max( [ preP['indsToCalculateReceptiveField'] + indPoolForTrain ][0] ) # we'll throw out unused samples.

## 2. Pre-processing parameters for the thumbnails:
preP['downsampleRate'] = 2
preP['crop'] = 2
preP['numFeatures'] =  85  # number of pixels in the receptive field
preP['pixelSum'] = 6
preP['showAverageImages'] = showAverageImages # boolean
preP['downsampleMethod'] = 0 # 0 means sum square patches of pixels. 1 means use bicubic interpolation.

preP['classLabels'] = classLabels # append
preP['useExistingConnectionMatrices'] = useExistingConnectionMatrices # boolean
preP['matrixParamsFilename'] = matrixParamsFilename

# generate the data array:
fA, activePixelInds, lengthOfSide = generateDownsampledMNISTSet(preP) # argin = preprocessingParams

pixNum, numPerClass, classNum = fA.shape
# The dataset fA is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
# fA = n x m x 10 array where n = #active pixels, m = #digits
#   from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.

#-----------------------------------

# Loop through the number of simulations specified:
print(f'starting sim(s) for goal = {goal}, trPerClass = {trPerClass}, '
	f'numSniffsPerSample = {numSniffs}')

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
	if showThumbnailsUsed:
		tempArray = np.zeros((lengthOfSide, numPerClass, classNum))
		tempArray[activePixelInds,:,:] = digitQueues
		normalize = 1
		titleStr = 'Input thumbnails'
		showFeatureArrayThumbnails(tempArray, showThumbnailsUsed, normalize, titleStr)

#    #-----------------------------------------

	print('NEXT STEPS')
	print('Step 1: Create a moth')
	print('Step 2: ...')
	print('Step 3: Greatness')

	# Create a moth. Either load an existing moth, or create a new moth
	if useExistingConnectionMatrices:
		pass
		# DEV NOTE: Implement this!!
		# # load 'matrixParamsFilename'
	else:
		# a) load template params
		modelParams = model_params

		# b) over-write default values below (if applicable)
		modelParams.nF = len(activePixelInds)
		modelParams.goal = goal

		# c) Now populate the moth's connection matrices using the modelParams
		modelParams = initializeConnectionMatrices(modelParams)

	modelParams.trueClassLabels = classLabels # misc parameter tagging along
	modelParams.saveAllNeuralTimecourses = saveAllNeuralTimecourses



#	# Define the experiment parameters, including book-keeping for time-stepped evolutions, eg
#    #       when octopamine occurs, time regions to poll for digit responses, windowing of Firing rates, etc
#    experimentParams = experimentFn( trClasses, classLabels, valPerClass )

#    #-----------------------------------
#
#    ## 3. run this experiment as sde time-step evolution:

#    simResults = sdeWrapper_fn( modelParams, experimentParams, digitQueues )
#
#    #-----------------------------------

#    ## Experiment Results: EN behavior, classifier calculations:
#
#    if ~isempty(saveResultsImageFolder)
#        if ~exist(saveResultsImageFolder)
#            mkdir(saveResultsImageFolder)
#        end
#    end
#    # Process the sim results to group EN responses by class and time:
#    r = viewENresponses_fn( simResults, modelParams, experimentParams, ...
#                                            showENPlots, classLabels, resultsFilename, saveResultsImageFolder )
#
#    # Calculate the classification accuracy:
#    # for baseline accuracy function argin, substitute pre- for post-values in r:
#    rNaive = r
#    for i = 1:length(r)
#        rNaive(i).postMeanResp = r(i).preMeanResp
#        rNaive(i).postStdResp = r(i).preStdResp
#        rNaive(i).postTrainOdorResp = r(i).preTrainOdorResp
#    end
#
#    # 1. Using Log-likelihoods over all ENs:
#    #     Baseline accuracy:
#    outputNaiveLogL = classifyDigitsViaLogLikelihood_fn ( rNaive )
#    # disp(  'LogLikelihood: ')
#        disp( [ 'Naive  Accuracy: ' num2str(round(outputNaiveLogL.totalAccuracy)),...
#         '#, by class: ' num2str(round(outputNaiveLogL.accuracyPercentages)),    ' #.   ' ])

#    #    Post-training accuracy using log-likelihood over all ENs:
#    outputTrainedLogL = classifyDigitsViaLogLikelihood_fn ( r )
#    disp([ 'Trained Accuracy: ' num2str(round(outputTrainedLogL.totalAccuracy)),...
#        '#, by class: ' num2str(round(outputTrainedLogL.accuracyPercentages)),    ' #.   '  resultsFilename, '_', num2str(run) ])

#    # 2. Using single EN thresholding:
#    outputNaiveThresholding = classifyDigitsViaThresholding_fn ( rNaive, 1e9, -1, 10 )
#    outputTrainedThresholding = classifyDigitsViaThresholding_fn ( r, 1e9, -1, 10 )
##     disp( 'Thresholding: ')
##     disp( [ 'Naive accuracy: ' num2str(round(outputNaiveThresholding.totalAccuracy)),...
##               '#, by class: ' num2str(round(outputNaiveThresholding.accuracyPercentages)),    ' #.   ' ])
##     disp([ ' Trained accuracy: ' num2str(round(outputTrainedThresholding.totalAccuracy)),...
##               '#, by class: ' num2str(round(outputTrainedThresholding.accuracyPercentages)),    ' #.   ' ])

#    # append the accuracy results, and other run data, to the first entry of r:
#    r(1).modelParams = modelParams  # will include all connection weights of this moth
#    r(1).outputNaiveLogL = outputNaiveLogL
#    r(1).outputTrainedLogL = outputTrainedLogL
#    r(1).outputNaiveThresholding = outputNaiveThresholding
#    r(1).outputTrainedThresholding = outputTrainedThresholding
#    r(1).matrixParamsFilename = matrixParamsFilename
#    r(1).K2Efinal = simResults.K2Efinal

#    if ~isempty(saveResultsDataFolder)
#        if ~exist(saveResultsDataFolder, 'dir' )
#            mkdir(saveResultsDataFolder)
#        end
#        save( fullfile(saveResultsDataFolder, [resultsFilename, '_', num2str(run) ]) , 'r')
#    end

#end # for run
