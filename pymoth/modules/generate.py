#!/usr/bin/env python3

"""

.. module:: generate
   :platform: Unix
   :synopsis: Download (if absent) and prepare down-sampled MNIST dataset.

.. moduleauthor:: Adam P. Jones <ajones173@gmail.com>

"""

import numpy as _np
import os as _os
from skimage.transform import downscale_local_mean

def generate_ds_mnist( max_ind, class_labels, crop, downsample_ratio, downsample_method,
inds_to_ave, pixel_sum, inds_to_calc_RF, num_features, screen_size, save_results_folder,
show_thumbnails, data_dir='/tmp', data_fname='MNIST_all'):
	"""
	Preprocessing:
		#. Load MNIST
		#. cropping and downsampling
		#. mean-subtract, make non-negative, normalize pixel sums
		#. select active pixels (receptive field)

	Loads the MNIST dataset (from Yann LeCun's website), then applies various \
	preprocessing steps to reduce the number of pixels (each pixel will be a feature).

	The 'receptive field' step destroys spatial relationships, so to reconstruct \
	a 12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices \
	can be embedded in a 144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.

	Args:
		max_ind (int): maximum number of samples to use
		class_labels (numpy array): numeric classes (for MNIST, digits 0:9)
		crop (int): image cropping parameter
		downsample_ratio (int): image downsample ratio (n:1)
		downsample_method (int): method for downsampling image
		inds_to_ave (numpy array): pixel indices
		pixel_sum (int): normalization factor
		inds_to_calc_RF (numpy array): pixel indices for receptive field
		num_features (int): number of pixels in the receptive field
		screen_size (tuple): screen size (width, height) for images
		save_results_folder (str): absolute path to where results will be saved
		show_thumbnails (int): number of thumbnails to show for each class (0 means none)
		data_dir (str): optional keyword arg specifying where to save data
		data_fname (str): optional keyword arg specifying filename of saved data

	Returns
	-------
		feature_array (numpy array)
			feature array [#active pixels x #samples x #classes]
		active_pixel_inds (list)
			pixel indices to allow thumbnail viewing
		len_side (int)
			allows reconstruction of thumbnails given from the feature vectors

	>>> generate_ds_mnist(
					  max_ind,
					  class_labels,
					  crop,
					  downsample_ratio,
					  downsample_method,
					  [i for i in range(550,1000)],
					  6,
					  [i for i in range(550,1000)],
					  85,
					  screen_size,
					  '',
					  0,
					 )

	"""

	# if data_dir specified (not the default value), prepend home dir path
	if data_dir!='/tmp':
		data_dir = _os.path.expanduser("~")+_os.sep+data_dir

	##TEST for existence of data folder, else create it
	if not _os.path.isdir(data_dir):
		_os.mkdir(data_dir)
		print('\nCreating data directory: {}\n'.format(data_dir))

	mnist_fpath = data_dir + _os.sep + data_fname + '.npy'

	# test for npy file before loading. run creation script, if absent.
	if not _os.path.isfile(mnist_fpath):
		# download and save data from the web
		from ..MNIST_all import MNIST_make_all
		MNIST_make_all.make_MNIST(mnist_fpath)

	# 1. extract mnist:
	mnist = _np.load(mnist_fpath, allow_pickle = True).item()
	# loads dictionary 'mnist' with keys:value pairs =
	# .train_images, .test_images, .train_labels, .test_labels (ie the original data from PMTK3)
	# AND parsed by class. These fields are used to assemble the image_array:
	# .trI_* = train_images of class *
	# .teI_* = test_images of class *
	# .trL_* = train_labels of class *
	# .teL_* = test_labels of class *

	# extract the required images and classes
	image_indices = range(max_ind+1)
	image_array = extract_mnist_feature_array(mnist, class_labels, image_indices, 'train')
	# image_array = numberImages x h x w x numberClasses 4-D array. class order: 1 to 10 (10 = '0')

	# calc new dimensions
	im_z, im_height, im_width, label_len = image_array.shape
	crop_val = crop*_np.ones(4,dtype = int)
	new_width = (im_width-_np.sum(crop_val[2:]))/downsample_ratio
	new_height = (im_height-_np.sum(crop_val[0:2]))/downsample_ratio
	new_length = int(new_width*new_height)

	feature_array = _np.zeros((new_length, im_z, label_len)) # pre-allocate

	# crop, downsample, and vectorize the average images and the image stacks
	for c in range(label_len):
		# feature_array[...,n] : [a x numImages] array,
		# 	where a = number of pixels in the cropped and downsampled images
		feature_array[...,c] = crop_downsample_vectorize_images(image_array[...,c],
			crop, downsample_ratio, downsample_method)

	del image_array # to save memory

	# subtract a mean image from all feature vectors, then make values non-negative

	# a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
	overall_ave = _np.zeros((new_length, )) # pre-allocate col vector
	class_ave_raw = _np.zeros((new_length, label_len))
	for c in range(label_len):
		class_ave_raw[:,c] = average_image_stack(feature_array[:, inds_to_ave, c],
			list(range(len(inds_to_ave))) )
		overall_ave += class_ave_raw[:,c]
	overall_ave /= label_len

	# b. Subtract this overall_ave image from all images
	ave_2D = _np.tile(overall_ave,(im_z,1)).T
	ave_3D = _np.repeat(ave_2D[:,:,_np.newaxis],label_len,2)
	feature_array -= ave_3D
	del ave_2D, ave_3D

	feature_array = _np.maximum(feature_array, 0) # remove any negative pixel values

	# c. Normalize each image so the pixels sum to the same amount
	f_sums = _np.sum(feature_array, axis=0)
	norm_array = _np.repeat(f_sums[_np.newaxis,:,:],new_length,0)
	feature_array *= pixel_sum
	feature_array /= norm_array
	# feature_array now consists of mean-subtracted, non-negative,
	# normalized (by sum of pixels) columns, each column a vectorized thumbnail.
	# size = 144 x numDigitsPerClass x 10

	len_side = new_length # save to allow sde_EM_evolution to print thumbnails.

	# d. Define a Receptive Field, ie the active pixels
	# Reduce the number of features by getting rid of less-active pixels.
	# If we are using an existing moth then active_pixel_inds is already defined, so
	# we need to load the modelParams to get the number of features
	# (since this is defined by the AL architecture):

	# reduce pixel number (downsample) to reflect # of features in moth brain
	fA_sub = feature_array[:, inds_to_calc_RF, :]
	active_pixel_inds = select_active_pixels(fA_sub, num_features,
		screen_size, save_image_folder=save_results_folder,
		show_thumbnails=show_thumbnails)
	feature_array = feature_array[active_pixel_inds,:,:].squeeze() # Project onto the active pixels

	return feature_array, active_pixel_inds, len_side

def extract_mnist_feature_array(mnist, labels, image_indices, phase_label):
	"""

	Extracts a subset of the samples from each class, converts the images to doubles \
	on [0 1], and returns a 4-D array.

	Args:
		mnist (dict): loaded from `MNIST_all.npy`
		labels (numpy array): numeric classes (for MNIST, digits 0:9)
		image_indices (range): images you want from each class
		phase_label (str): Image set to draw from ('train' or 'test')

	Returns
	-------
		im_array (numpy array)
			4-D array [#images x image_height x image_width x #classes]

	>>> image_array = extract_mnist_feature_array(mnist, class_labels, \
	range(max_ind+1), 'train')

	"""

	# get some dimensions:
	(h,w) = mnist['train_images'].shape[1:3]
	max_ind = max(image_indices)

	# initialize outputs:
	im_array = _np.zeros((max_ind+1, h, w, len(labels)))

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

def crop_downsample_vectorize_images(im_stack, crop_val, downsample_ratio, downsample_method):
	"""

	For each image in a stack of images; crop, downsample, then make into a vector.

	Args:
		im_stack (numpy array): [numImages x width x height]
		crop_val: number of pixels to shave off each side. (int) or (list) [top, \
		bottom, left, right]
		downsample_ratio (int): image downsample ratio (n:1)
		downsample_method (int): method for downsampling image (0: sum square patches, \
		1: bicubic interpolation)

	Returns
	-------
		im_array (numpy array)
			[#pixels x #images] array, where #pixels refers to the number of \
			pixels in the cropped and downsampled images.

	>>> crop_downsample_vectorize_images(dummy_image_array[...,0],2,2,1)

	"""

	if type(crop_val) is int:
		crop_val = crop_val*_np.ones(4,dtype = int)

	if len(im_stack.shape)==3:
		im_z,im_height,im_width = im_stack.shape
	else:
		im_height,im_width = im_stack.shape
		im_z = 1

	width = range(crop_val[2], im_width-crop_val[3])
	height = range(crop_val[0], im_height-crop_val[1])

	new_width = (im_width-_np.sum(crop_val[2:]))/downsample_ratio
	new_height = (im_height-_np.sum(crop_val[0:2]))/downsample_ratio

	im_col_array = _np.zeros((int(new_width*new_height),im_z))
	# crop, downsample, vectorize the thumbnails one-by-one
	for s in range(im_z):
		t = im_stack[s,...]
		# crop image
		ixgrid = _np.ix_(width, height)
		t = t[ixgrid]

		if downsample_method: # bicubic
			t2 = downscale_local_mean(t, (downsample_ratio, downsample_ratio))

		else: # sum 2 x 2 blocks
			t2 = _np.zeros((int(len(height)/downsample_ratio),int(len(width)/downsample_ratio)))
			for i in range(int(len(height)/downsample_ratio)):
				for j in range(int(len(width)/downsample_ratio)):
					b = t[(i-1)*downsample_ratio+1:i*downsample_ratio+1,
						(j-1)*downsample_ratio+1:j*downsample_ratio+1]
					t2[i,j] = b.sum()

		im_col_array[:,s] = t2.flatten()/t2.max()

	return im_col_array

def average_image_stack( im_stack, indices_to_average ):
	"""

	Average a stack of images.

	Args:
		im_stack (numpy array): 3-d stack (x, y, z) OR 2-d matrix (images-as-col-vecs, z)
		indices_to_average (list): which images in the stack to average

	Returns
	-------
		average_image (numpy array)
			average of image stack.

	>>> average_im = average_image_stack(dummy_feature_array[...,0], list(range(5)))

	"""

	im_stack_shape = im_stack.shape

	# case: images are col vectors
	if len(im_stack_shape) == 2:
		ave_im = _np.zeros((im_stack_shape[0],))
	else:
		ave_im = _np.zeros(im_stack_shape)

	for i in indices_to_average:
		ave_im += im_stack[:, i]

	# normalize
	ave_im /= im_stack_shape[1]

	return ave_im

def select_active_pixels( feature_array, num_features, screen_size, save_image_folder=[], show_thumbnails=0 ):
	"""
	Select the most active pixels, considering all class average images, to use as features.

	Args:
		feature_array (numpy array): 3-D array # of features X # samples per class X \
		# of classes, created by :func:`generate_ds_mnist`.
		num_features (int): number of pixels in the receptive field
		save_image_folder (str): directory to save average thumbnail images (if \
		empty, don't save)
		screen_size (tuple): screen size (width, height) for images
		show_thumbnails (int): number of thumbnails to plot

	Returns
		active_pixel_inds (numpy array)
			1 x nF vector of indices to use as features. Indices are relative to \
			the vectorized thumbnails (so between 1 and 144).

	>>> active_pixel_inds = select_active_pixels(feature_array, 85, (1920, 1080))

	"""

	## make classAves matrix (cA)
	# each col a class ave 1 to 10 (ie 0), and add a col for the overall_ave
	num_pix, num_per_class, num_classes  = feature_array.shape
	cA = _np.zeros((num_pix, num_classes+1))

	for i in range(num_classes):
		cA[:,i] = average_image_stack(feature_array[:,:,i], list(range(num_per_class)))

	# last col = average image over all digits
	cA[:,-1] = _np.sum(cA[:,:-1], axis=1) / num_classes

	# normed version (does not rescale the overall average)
	z = _np.max(cA, axis=0)
	z[-1] = 1
	cA_norm = cA/_np.tile(z, (num_pix,1))

	# select most active 'num_features' pixels
	peak_pix = cA[:, :-1]
	peak_pix_logical = _np.zeros(peak_pix.shape)

	# all the pixel values from all the class averages, in descending order
	vals = _np.sort(peak_pix.flatten())[::-1]

	# start selecting the highest-valued pixels
	stop = 0
	while not stop:
		thresh = vals.max()
		peak_pix_logical[peak_pix>=thresh] = 1
		active_pix = peak_pix_logical.sum(axis=1) # sum the rows
		# If a class ave had the i'th pixel, selected, keptPixels(i) > 0
		stop = (active_pix > 0).sum() >= num_features # check if we have enough pixels

		vals = vals[vals < thresh]  # peel off the value(s) just used

	active_pixel_inds = _np.nonzero(active_pix > 0)[0]

	if show_thumbnails and save_image_folder:

		from ..modules.show_figs import show_FA_thumbs
		# plot the normalized classAves pre-ablation
		normalize = 0
		title_str = 'class aves, all pixels'
		show_FA_thumbs(cA_norm, num_classes+1, normalize, title_str,
			screen_size,
			save_image_folder + _os.sep + 'thumbnails_all')

		# look at active pixels of the classAves, ie post-ablation
		normalize = 0
		cA_active_only = _np.zeros(cA_norm.shape)
		cA_active_only[active_pixel_inds, : ] = cA_norm[active_pixel_inds, :]
		title_str = 'class aves, active pixels only'
		show_FA_thumbs(cA_active_only, num_classes+1, normalize, title_str,
			screen_size, save_image_folder + _os.sep + 'thumbnails_active')

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
