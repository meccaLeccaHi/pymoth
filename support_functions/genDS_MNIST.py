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
	fa_shape = featureArray.shape
	print('this featureArray shape:', fa_shape)

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

	print(overallAve.shape)
	print(z,h,w,label_len)

	print('foo')
	foo = np.array([np.tile(overallAve, (1,z)) for i in range(label_len)])
	([np.tile(overallAve, (m,n)) for i in xrange(p)])
	print(foo.shape)


	#featureArray -= np.tile(overallAve, (1, z, label_len))

	# b. Subtract this overallAve image from all images:
	#featureArray = featureArray - repmat( overallAve, [1, size(featureArray,2), size(featureArray,3) ] );
	#featureArray = max( featureArray, 0 );  kill negative pixel values

	# c. Normalize each image so the pixels sum to the same amount:
	#fSums = sum(featurpreP['maxInd']eArray,1);
	#fNorm = preP.pixelSum*featureArray./repmat(fSums, [size(featureArray,1), 1, 1 ] );
	#featureArray = fNorm;
	# featureArray now consists of mean-subtracted, non-negative,
	# normalized (by sum of pixels) columns, each column a vectorized thumbnail. size = 144 x numDigitsPerClass x 10

	#lengthOfSide = size(featureArray,1);  save to allow sde_EM_evolution to print thumbnails.

	# d. Define a Receptive Field, ie the active pixels:
	# Reduce the number of features by getting rid of less-active pixels.
	# If we are using an existing moth then activePixelInds is already defined, so
	# we need to load the modelParams to get the number of features (since this is defined by the AL architecture):
	#if preP.useExistingConnectionMatrices
	#    load( preP.matrixParamsFilename );     loads 'modelParams'
	#    preP.numFeatures = modelParams.nF;
	#end
	#activePixelInds = selectActivePixels_fn( featureArray( :, preP.indsToCalculateReceptiveField, : ),...
	#                                                                                  preP.numFeatures, preP.showAverageImages );
	#featureArray = featureArray(activePixelInds,:,:);    Project onto the active pixels
