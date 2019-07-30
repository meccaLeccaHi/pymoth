#!/usr/bin/env python3

import numpy as np
import os

def generate_ds_MNIST( max_ind, class_labels, crop, downsample_rate, downsample_method,
inds_to_ave, pixel_sum, inds_to_calc_RF, num_features, screen_size, save_results_folder,
show_thumbnails ):

	'''
	Loads the MNIST dataset (from Yann LeCun's website),
	then applies various preprocessing steps to reduce the number of pixels
	(each pixel will be a feature).

	The 'receptive field' step destroys spatial relationships, so to reconstruct
	a 12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices
	can be embedded in a 144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.
	Modify the path for the MNIST data file as needed.

	Inputs (preprocessing parameters):
		max_ind, class_labels, crop,

		downsample_rate: downsample ratio
		downsample_method: method for downsampling image

		inds_to_ave, pixel_sum, inds_to_calc_RF,

		num_features: number of pixels in the receptive field
		screen_size: screen size (width, height) for images
		save_results_folder: absolute path to where results will be saved
		show_thumbnails: number of thumbnails to show for each class (0 means none)

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
	Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
	MIT License
	'''

	im_dir = 'MNIST_all'

	mnist_fname = os.path.dirname(os.path.dirname(__file__)) + os.sep + im_dir + os.sep + 'MNIST_all.npy'

	# test for npy file before loading. run creation script, if absent.
	if not os.path.isfile(mnist_fname):
		# download and save data from the web
		from MNIST_all import MNIST_makeAll
		MNIST_makeAll.download_save()

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
	imageIndices = range(max_ind+1)
	imageArray = extractMNISTFeatureArray(mnist, class_labels, imageIndices, 'train')
	# imageArray = numberImages x h x w x numberClasses 4-D array. class order: 1 to 10 (10 = '0')

	# calc new dimensions
	im_z, im_height, im_width, label_len = imageArray.shape
	cropVal = crop*np.ones(4,dtype = int)
	new_width = (im_width-np.sum(cropVal[2:]))/downsample_rate
	new_height = (im_height-np.sum(cropVal[0:2]))/downsample_rate
	new_length = int(new_width*new_height)

	featureArray = np.zeros((new_length, im_z, label_len)) # pre-allocate

	# crop, downsample, and vectorize the average images and the image stacks
	for c in range(label_len):
		# featureArray[...,n] : [a x numImages] array,
		# 	where a = number of pixels in the cropped and downsampled images
		featureArray[...,c] = cropDownsampleVectorizeImageStack(imageArray[...,c],
		 	crop, downsample_rate, downsample_method)

	del imageArray # to save memory

	# subtract a mean image from all feature vectors, then make values non-negative

	# a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
	overallAve = np.zeros((new_length, )) # pre-allocate col vector
	classAvesRaw = np.zeros((new_length, label_len))
	for c in range(label_len):
		classAvesRaw[:,c] = averageImageStack(featureArray[:, inds_to_ave, c],
			list(range(len(inds_to_ave))) )
		overallAve += classAvesRaw[:,c]
	overallAve /= label_len

	# b. Subtract this overallAve image from all images
	ave_2D = np.tile(overallAve,(im_z,1)).T
	ave_3D = np.repeat(ave_2D[:,:,np.newaxis],label_len,2)
	featureArray -= ave_3D
	del ave_2D, ave_3D

	featureArray = np.maximum( featureArray, 0 ) # remove any negative pixel values

	# c. Normalize each image so the pixels sum to the same amount
	fSums = np.sum(featureArray, axis=0)
	normArray = np.repeat(fSums[np.newaxis,:,:],new_length,0)
	featureArray *= pixel_sum
	featureArray /= normArray
	# featureArray now consists of mean-subtracted, non-negative,
	# normalized (by sum of pixels) columns, each column a vectorized thumbnail.
	# size = 144 x numDigitsPerClass x 10

	lengthOfSide = new_length # save to allow sde_EM_evolution to print thumbnails.

	# d. Define a Receptive Field, ie the active pixels
	# Reduce the number of features by getting rid of less-active pixels.
	# If we are using an existing moth then activePixelInds is already defined, so
	# we need to load the modelParams to get the number of features
	# (since this is defined by the AL architecture):

	# reduce pixel number (downsample) to reflect # of features in moth brain
	fA_sub = featureArray[:, inds_to_calc_RF, :]
	activePixelInds = selectActivePixels(fA_sub, num_features,
		screen_size, save_image_folder=save_results_folder,
		show_thumbnails=show_thumbnails)
	featureArray = featureArray[activePixelInds,:,:].squeeze() # Project onto the active pixels

	return featureArray, activePixelInds, lengthOfSide

def extractMNISTFeatureArray( mnist, labels, image_indices, phase_label ):
    '''
    Extract a subset of the samples from each class, convert the images to doubles on [0 1], and
        return a 4-D array: 1, 2 = im. 3 indexes images within a class, 4 is the class.

    Inputs:
        mnist = dict loaded by 'MNIST_all.npy'
            with fields = training_images, test_images, training_labels, test_labels
        trI = mnist['train_images']
        teI = mnist['test_images']
        trL = mnist['train_labels']
        teL = mnist['test_labels']
        labels = vector of the classes (digits) you want to extract
        image_indices = list of which images you want from each class
        phase_label = 'train' or 'test'. Determines which images you draw from
            (since we only need a small subset, one or the other is fine)

    Outputs:
        im_array = numberImages x h x w x numberClasses 4-D array

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    # get some dimensions:
    (h,w) = mnist['train_images'].shape[1:3]
    max_ind = max(image_indices)

    # initialize outputs:
    im_array = np.zeros((max_ind+1, h, w, len(labels)))

    # process each class in turn:
    for c in labels:
        if phase_label=='train': # 1 = extract train, 0 = extract test
            im_data = mnist['train_images']
            target_data = mnist['train_labels']
        else:
            im_data = mnist['test_images']
            target_data = mnist['test_labels']

        # Convert from (8-bit) unsigned integers to double precision float
        #  see: (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html)
        class_array = im_data[target_data==c].astype('float64')/256

        im_array[image_indices,:,:,c] = class_array[image_indices,:,:]

    return im_array

def cropDownsampleVectorizeImageStack( im_stack, crop_val, ds_ratio, ds_method ):
    '''
    For each image in a stack of images: Crop, then downsample, then make into a col vector.
    Inputs:
        1. im_stack: numImages x width x height array
        2. crop_val: number of pixels to shave off each side. can be a scalar or a
            4 x 1 vector: top, bottom, left, right.
        3. ds_ratio: amount to downsample
        4. ds_method: if 0, do downsampling by summing square patches.
            If 1, use bicubic interpolation.
    Returns:
        1. im_array: a x numImages array, where a = number of pixels in the cropped and downsampled images

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    from scipy.misc import imresize

    if type(crop_val) is int:
        crop_val = crop_val*np.ones(4,dtype = int)

    if len(im_stack.shape)==3:
        im_z,im_height,im_width = im_stack.shape
    else:
        im_height,im_width = im_stack.shape
        im_z = 1

    width = range(crop_val[2], im_width-crop_val[3])
    height = range(crop_val[0], im_height-crop_val[1])

    new_width = (im_width-np.sum(crop_val[2:]))/ds_ratio
    new_height = (im_height-np.sum(crop_val[0:2]))/ds_ratio

    im_col_array = np.zeros((int(new_width*new_height),im_z))
    # crop, downsample, vectorize the thumbnails one-by-one
    for s in range(im_z):
        t = im_stack[s,...]
        # crop image
        ixgrid = np.ix_(width, height)
        t = t[ixgrid]

        if ds_method: # bicubic
            t2 = imresize(t, 1/ds_ratio, interp='bicubic')

        else: # sum 2 x 2 blocks
            t2 = np.zeros((int(len(height)/ds_ratio),int(len(width)/ds_ratio)))
            for i in range(int(len(height)/ds_ratio)):
                for j in range(int(len(width)/ds_ratio)):
                    b = t[(i-1)*ds_ratio+1:i*ds_ratio+1, (j-1)*ds_ratio+1:j*ds_ratio+1]
                    t2[i,j] = b.sum()

        im_col_array[:,s] = t2.flatten()/t2.max()

    return im_col_array

def averageImageStack( im_stack, indices_to_average ):
    '''
    Average a stack of images
    Inputs:
        1. im_stack = 3-d stack (x, y, z) OR 2-d matrix (images-as-col-vecs, z)
        Caution: Do not feed in feature_array (ie 3-d with dim 1 = feature cols, 2 = samples per class, 3 = classes)
        2. indices_to_average: which images in the stack to average
    Returns:
        1. average_image: (if input is 3-d) or column vector (if input is 2-d)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    im_stack_shape = im_stack.shape

    # case: images are col vectors
    if len(im_stack_shape) == 2:
        ave_im = np.zeros((im_stack_shape[0],))
    else:
        ave_im = np.zeros(im_stack_shape)

    for i in indices_to_average:
        ave_im += im_stack[:, i]

    # normalize
    ave_im /= im_stack_shape[1]

    return ave_im

def selectActivePixels( feature_array, num_features, screen_size,
    save_image_folder=[], show_thumbnails=0 ):
    '''
    Select the most active pixels, considering all class average images, to use as features.
    Inputs:
        1. feature_array: 3-D array nF x nS x nC, where nF = # of features,
        nS = # samples per class, nC = number of classes. As created by genDS_MNIST.
        2. num_features: The number of active pixels to use (these form the receptive field).
        3. save_image_folder: dir to save average class images, empty = don't save
        4. screensize: (width, height)
        5. show_thumbnails: number of thumbnails to plot
    Returns:
        1. active_pixel_inds: 1 x nF vector of indices to use as features.
        Indices are relative to the vectorized thumbnails (so between 1 and 144).

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    # make a classAves matrix (cA), each col a class ave 1 to 10 (ie 0),
    #  and add a col for the overallAve
    from modules.show_figs import show_FA_thumbs

    num_pix, num_per_class, num_classes  = feature_array.shape
    cA = np.zeros((num_pix, num_classes+1))

    for i in range(num_classes):
        cA[:,i] = averageImageStack(feature_array[:,:,i], list(range(num_per_class)))

    # last col = average image over all digits
    cA[:,-1] = np.sum(cA[:,:-1], axis=1) / num_classes

    # normed version (does not rescale the overall average)
    z = np.max(cA, axis=0)
    z[-1] = 1
    cA_norm = cA/np.tile(z, (num_pix,1))

    # select most active 'num_features' pixels
    peak_pix = cA[:, :-1]
    peak_pix_logical = np.zeros(peak_pix.shape)

    # all the pixel values from all the class averages, in descending order
    vals = np.sort(peak_pix.flatten())[::-1]

    # start selecting the highest-valued pixels
    stop = 0
    while not stop:
        thresh = vals.max()
        peak_pix_logical[peak_pix>=thresh] = 1
        active_pix = peak_pix_logical.sum(axis=1) # sum the rows
        # If a class ave had the i'th pixel, selected, keptPixels(i) > 0
        stop = (active_pix > 0).sum() >= num_features # check if we have enough pixels

        vals = vals[vals < thresh]  # peel off the value(s) just used

    active_pixel_inds = np.nonzero(active_pix > 0)[0]

    if show_thumbnails and save_image_folder:
        # plot the normalized classAves pre-ablation
        normalize = 0
        title_str = 'class aves, all pixels'
        show_FA_thumbs(cA_norm, num_classes+1, normalize, title_str,
            screen_size,
            save_image_folder + os.sep + 'thumbnails_all')

        # look at active pixels of the classAves, ie post-ablation
        normalize = 0
        cA_active_only = np.zeros(cA_norm.shape)
        cA_active_only[active_pixel_inds, : ] = cA_norm[active_pixel_inds, :]
        title_str = 'class aves, active pixels only'
        show_FA_thumbs(cA_active_only, num_classes+1, normalize, title_str,
            screen_size, save_image_folder + os.sep + 'thumbnails_active')

    return active_pixel_inds

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
