def generateDownsampledMNISTSet( preP ):
	# Loads the MNIST dataset (from Yann LeCun's website),
	# then applies various preprocessing steps to reduce the number of pixels (each pixel will
	# be a feature).
	# The 'receptive field' step destroys spatial relationships, so to reconstruct a
	# 12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices can be embedded in a
	# 144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.
	# Modify the path for the MNIST data file as needed.
	#
	# Inputs:
	#   1. preP = preprocessingParams = dictionary with keys corresponding to relevant variables
	#
	# Outputs:
	#   1. featureArray = n x m x 10 array. n = #active pixels, m = #digits from each class that
	#	will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.
	#   2. activePixelInds: list of pixel indices to allow re-embedding into empty thumbnail for viewing.
	#   3. lengthOfSide: allows reconstruction of thumbnails given from the  feature vectors.
	#------------------------------------------------------------------
	# Preprocessing includes:
	#   1. Load MNIST set.generateDownsampledMnistSet_fn
	#   2. cropping and downsampling
	#   3. mean-subtract, make non-negative, normalize pixel sums
	#   4. select active pixels (receptive field)

	import os
	import numpy as np
	from support_functions.extractFA import extractMNISTFeatureArray
	from support_functions.vec_images import cropDownsampleVectorizeImageStack
	from support_functions.aveImStack import averageImageStack
	from support_functions.selectPix import selectActivePixels
	# DEV NOTE: Collapse these babies into one beautiful object

	im_dir = 'MNIST_all'

	# 1. extract mnist:
	mnist = np.load(os.path.join(im_dir,'MNIST_all.npy')).item()

	# loads dictionary 'mnist' with keys:value pairs =
	#              .training_images, .test_images, .training_labels, .test_labels (ie the original data from PMTK3)
	#              AND parsed by class. These fields are used to assemble the imageArray:
	#              .trI_* = train_images of class *;
	#              .teI_* = test_images of class *;
	#              .trL_* = train_labels of class *;
	#              .teL_* = test_labels of class *;

	# extract the required images and classes
	imageIndices = range(preP['maxInd']+1)
	imageArray = extractMNISTFeatureArray(mnist, preP['classLabels'], imageIndices, 'train')
	# imageArray = numberImages x h x w x numberClasses 4-D array. class order: 1 to 10 (10 = '0')

	z,h,w,label_len = imageArray.shape
	# DEV NOTE: This is hard code now :( - fix this during refactor
	new_length = 144

	featureArray = np.zeros((new_length, z, label_len)) # pre-allocate

	# crop, downsample, and vectorize the average images and the image stacks
	for c in range(label_len):
		# featureMatrix : [a x numImages] array,
		# 	where a = number of pixels in the cropped and downsampled images
		featureMatrix = cropDownsampleVectorizeImageStack(imageArray[...,c],
		 	preP['crop'], preP['downsampleRate'], preP['downsampleMethod'])
		featureArray[...,c] = featureMatrix

	del imageArray   # to save memory

	# subtract a mean image from all feature vectors, then make values non-negative

	# DEV NOTE: The following loop could be collapsed into the loop above
	# a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
	overallAve = np.zeros((new_length, )) # pre-allocate col vector
	classAvesRaw = np.zeros((new_length, label_len))
	for c in range(label_len):
		classAvesRaw[:,c] = averageImageStack(featureArray[:, preP['indsToAverageGeneral'], c],
			list(range(len(preP['indsToAverageGeneral']))) )
		overallAve += classAvesRaw[:,c]
	overallAve /= label_len

	# b. Subtract this overallAve image from all images
	ave_2D = np.tile(overallAve,(z,1)).T
	# ave_2D = npm.repmat(overallAve,z,1).T
	ave_3D = np.repeat(ave_2D[:,:,np.newaxis],label_len,2)
	featureArray -= ave_3D
	del ave_2D, ave_3D

	featureArray = featureArray.clip(min=0) # remove any negative pixel values

	# c. Normalize each image so the pixels sum to the same amount
	fSums = np.sum(featureArray, axis=0)
	normArray = np.repeat(fSums[np.newaxis,:,:],new_length,0)
	fNorm = preP['pixelSum']*featureArray/normArray
	featureArray = fNorm
	## DEV NOTE: Replace above with syntax below
	## featureArray *= preP['pixelSum']
	## featureArray /= normArray
	# featureArray now consists of mean-subtracted, non-negative,
	# normalized (by sum of pixels) columns, each column a vectorized thumbnail. size = 144 x numDigitsPerClass x 10

	lengthOfSide = new_length # save to allow sde_EM_evolution to print thumbnails.

	# d. Define a Receptive Field, ie the active pixels
	# Reduce the number of features by getting rid of less-active pixels.
	# If we are using an existing moth then activePixelInds is already defined, so
	# we need to load the modelParams to get the number of features (since this is defined by the AL architecture):
	if preP['useExistingConnectionMatrices']:
		pass
		# DEV NOTE: Implement this!!
		# # load 'modelParams'
		# load( preP['matrixParamsFilename'] )
		# preP['numFeatures'] = modelParams['nF']

	# DEV NOTE: Clarify this part with CBD - need to understand 'active pixels' better
	fA_sub = featureArray[:, preP['indsToCalculateReceptiveField'], :]
	activePixelInds = selectActivePixels(fA_sub, preP['numFeatures'], preP['showAverageImages'])
	featureArray = featureArray[activePixelInds,:,:].squeeze() # Project onto the active pixels

	return featureArray, activePixelInds, lengthOfSide
