def setMNISTExperimentParams( trClasses, classLabels, valPerClass ):
	# This function defines parameters of a time-evolution experiment: overall timing, stim timing and
	# strength, octo timing and strength, lowpass window parameter, etc.
	# It does book-keeping to allow analysis of the SDE time-stepped evolution of the neural firing rates.
	# Inputs:
	#	1. trClasses: vector of indices giving the classes of the training digits in order.
	#		The first entry must be nonzero. Unused entries can be filled with -1s if wished.
	# 	2. classLabels: a list of labels, eg 1:10 for mnist
	# 	3. valPerClass: how many digits of each class to use for baseline and post-train
	# Output:
	#	1. expParams: struct with experiment info.

#-------------------------------------------------------------------------------

	# Order of time periods:
	#	1. no event period: allow system to settle to a steady state spontaneous FR baseline
	#   2. baseline period: deliver a group of digits for each class
	#	3. no event buffer
	#	4. training period:  deliver digits + octopamine + allow hebbian updates
	#	5. no event buffer
	#	6. post-training period: deliver a group of digits for each class

	# import packages
	import numpy as np

	stimMag = 20 # stim magnitudes as passed into AL (See original version in smartAsABug codebase)
	stimLength = 0.22
	nC = len(classLabels) # the number of classes in this experiment

	## Define the time span and events:
	step = 3 # the time between digits (3 seconds)
	trStep = step + 2 # allow more time between training digits

	# DEV NOTE: The following hack allows us to set the attributes of a Python object
	# the same way we set the fields of a struct in matlab. Should be removed after
	# transition to OOP
	class Object(object):
		pass
	expParams = Object()

	expParams.simStart = -30 # use negative start-time for convenience (artifact)

	## Baseline period:
	# do a loop, to allow gaps between class groups:
	baselineTimes = []
	startTime = 30
	gap = 10

	for i in range(nC):
		# vector of timepoints
		baselineTimes += list(range(startTime, startTime + valPerClass*step, step))
		startTime = max(baselineTimes) + gap
	endOfBaseline = max(baselineTimes) + 25 # include extra buffer before training

	# Training period:
	# vector of timepoints, one digit every 'trStep' seconds
	trainTimes = list(range(endOfBaseline, endOfBaseline + len(trClasses)*trStep, trStep))
	endOfTrain = max(trainTimes) + 25 # includes buffer before Validation

	# Val period:
	# do a loop, to allow gaps between class groups
	valTimes = []
	startTime = endOfTrain
	for i in range(nC):
		# vector of timepoints
		valTimes += list(range(startTime, startTime + valPerClass*step, step))
		startTime = max(valTimes) + gap
	endOfVal = max(valTimes) + 4

	## assemble vectors of stimulus data for export:

	# Assign the classes of each stim. Assign the baseline and val in blocks,
	# and the training stims in the order passed in:
	whichClass = np.empty((1, len(baselineTimes+trainTimes+valTimes))) * np.nan
	numBaseline = valPerClass*nC
	numTrain = len(trClasses)
	for c in range(nC):
		# the baseline groups
		whichClass[ :, c*valPerClass : (c+1)*valPerClass ] = classLabels[c]

		# the val groups
		whichClass[ :, numBaseline + numTrain + c*valPerClass : \
			numBaseline + numTrain + (c+1)*valPerClass ]  = classLabels[c]

	whichClass[ :, numBaseline : numBaseline + numTrain ] = trClasses

	expParams.whichClass = whichClass

	stimStarts =  baselineTimes + trainTimes + valTimes
	expParams.stimStarts = np.array(stimStarts).reshape(1,-1) # starting times
	expParams.durations = stimLength*np.ones( (1,len(stimStarts)) ) # durations
	expParams.classMags = stimMag*np.ones( (1,len(stimStarts)) ) # magnitudes

	# octopamine input timing:
	expParams.octoMag = 1
	expParams.octoStart = trainTimes
	expParams.durationOcto = 1

	# Hebbian timing: Hebbian updates are enabled 25# of the way into the stimulus, and
	# last until 75% of the way through (ie active during the peak response period)
	expParams.hebStarts = [i + 0.25*stimLength for i in trainTimes]
	expParams.hebDurations = 0.5*stimLength*np.ones( (1,len(trainTimes)) )
	expParams.startTrain = min(expParams.hebStarts)
	expParams.endTrain = max(expParams.hebStarts) + max(expParams.hebDurations)

	## Other time parameters required for time evolution book-keeping:

	# the numbers 1,2,3 do refer to time periods where spont responses are
	# allowed to settle before recalibration.
	expParams.startPreNoiseSpontMean1 = -25
	expParams.stopPreNoiseSpontMean1 = -15
	# Currently no change is made in start/stopSpontMean2.
	# So spontaneous behavior may be stable in this range.
	expParams.startSpontMean2 = -10
	expParams.stopSpontMean2 = -5
	# currently, spontaneous behavior is steady-state by startSpontMean3.
	expParams.startSpontMean3 = 0
	expParams.stopSpontMean3 = 28

	expParams.preHebPollTime = min(trainTimes) - 5
	expParams.postHebPollTime = max(trainTimes) + 5

	# timePoints for plotting EN responses:
	# spontaneous response periods, before and after training, to view effect of
	# training on spontaneous FRs:
	expParams.preHebSpontStart = expParams.startSpontMean3
	expParams.preHebSpontStop = expParams.stopSpontMean3
	expParams.postHebSpontStart = max(trainTimes) + 5
	expParams.postHebSpontStop = min(valTimes) - 3

	# hamming filter window parameter (= width of transition zone in seconds).
	# The lp filter is applied to odors and to octo
	expParams.lpParam =  0.12

	expParams.simStop = max(stimStarts) + 10

	return expParams
