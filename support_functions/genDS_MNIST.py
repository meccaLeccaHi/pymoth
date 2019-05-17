def generateDownsampledMNISTSet( preP, saveImageFolder=[], scrsz = (1920, 1080) ):
	'''
	Loads the MNIST dataset (from Yann LeCun's website),
	then applies various preprocessing steps to reduce the number of pixels
	(each pixel will be a feature).

	The 'receptive field' step destroys spatial relationships, so to reconstruct
	a 12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices
	can be embedded in a 144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.
	Modify the path for the MNIST data file as needed.

	Inputs:
		1. preP = preprocessingParams = dictionary with keys corresponding to relevant variables
		2. saveImageFolder = image for thumbnails to be saved (if chosen)

	Outputs:
		1. featureArray = n x m x 10 array. n = #active pixels, m = #digits from
			each class that will be used.
			The 3rd dimension gives the class, 1:10 where 10 = '0'.
		2. activePixelInds: list of pixel indices to allow re-embedding into empty
			thumbnail for viewing.
	  	3. lengthOfSide: allows reconstruction of thumbnails given from the
			feature vectors.
	#---------------------------------------------------------------------------
	Preprocessing includes:
		1. Load MNIST set.generateDownsampledMnistSet_fn
		2. cropping and downsampling
		3. mean-subtract, make non-negative, normalize pixel sums
		4. select active pixels (receptive field)
	'''

	import os
	import numpy as np
	from support_functions.extractFA import extractMNISTFeatureArray
	from support_functions.vec_images import cropDownsampleVectorizeImageStack
	from support_functions.aveImStack import averageImageStack
	from support_functions.selectPix import selectActivePixels
	# DEV NOTE: Collapse these babies into one beautiful object

	im_dir = 'MNIST_all'

	mnist_fname = os.path.join(im_dir,'MNIST_all.npy')

	import pdb; pdb.set_trace()

	# test for npy file before loading. run creation script, if absent.
	if ~os.path.isfile(mnist_fname):
		# download and save data
		from MNIST_all import MNIST_makeAll
		MNIST_makeAll.downloadAndSave()

	# 1. extract mnist:
	mnist = np.load(mnist_fname, allow_pickle = True).item()
	# loads dictionary 'mnist' with keys:value pairs =
	#              .train_images, .test_images, .train_labels, .test_labels (ie the original data from PMTK3)
	#              AND parsed by class. These fields are used to assemble the imageArray:
	#              .trI_* = train_images of class *
	#              .teI_* = test_images of class *
	#              .trL_* = train_labels of class *
	#              .teL_* = test_labels of class *

	# extract the required images and classes
	imageIndices = range(preP['maxInd']+1)
	imageArray = extractMNISTFeatureArray(mnist, preP['classLabels'], imageIndices, 'train')
	# imageArray = numberImages x h x w x numberClasses 4-D array. class order: 1 to 10 (10 = '0')

	# calc new dimensions
	im_z, im_height, im_width, label_len = imageArray.shape
	cropVal = preP['crop']*np.ones(4,dtype = int)
	new_width = (im_width-np.sum(cropVal[2:]))/preP['downsampleRate']
	new_height = (im_height-np.sum(cropVal[0:2]))/preP['downsampleRate']
	new_length = int(new_width*new_height)

	featureArray = np.zeros((new_length, im_z, label_len)) # pre-allocate

	# crop, downsample, and vectorize the average images and the image stacks
	for c in range(label_len):
		# featureArray[...,n] : [a x numImages] array,
		# 	where a = number of pixels in the cropped and downsampled images
		featureArray[...,c] = cropDownsampleVectorizeImageStack(imageArray[...,c],
		 	preP['crop'], preP['downsampleRate'], preP['downsampleMethod'])

	del imageArray # to save memory

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
	ave_2D = np.tile(overallAve,(im_z,1)).T
	# ave_2D = npm.repmat(overallAve,im_z,1).T
	ave_3D = np.repeat(ave_2D[:,:,np.newaxis],label_len,2)
	featureArray -= ave_3D
	del ave_2D, ave_3D

	featureArray[ featureArray < 0 ] = 0 # remove any negative pixel values

	# c. Normalize each image so the pixels sum to the same amount
	fSums = np.sum(featureArray, axis=0)
	normArray = np.repeat(fSums[np.newaxis,:,:],new_length,0)
	featureArray *= preP['pixelSum']
	featureArray /= normArray
	# featureArray now consists of mean-subtracted, non-negative,
	# normalized (by sum of pixels) columns, each column a vectorized thumbnail.
	# size = 144 x numDigitsPerClass x 10

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
	activePixelInds = selectActivePixels(fA_sub, preP['numFeatures'], saveImageFolder,
		scrsz, preP['showThumbnails'])
	featureArray = featureArray[activePixelInds,:,:].squeeze() # Project onto the active pixels

	return featureArray, activePixelInds, lengthOfSide
