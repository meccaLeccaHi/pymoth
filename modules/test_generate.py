#!/usr/bin/env python3

import os
import numpy as np
from generate import generate_ds_mnist, extract_mnist_feature_array, \
    crop_downsample_vectorize_images, average_image_stack, select_active_pixels

# generate dummy data
mnist_fname = '.' + os.path.dirname(os.path.dirname(__file__)) + os.sep + \
    'MNIST_all' + os.sep + 'MNIST_all.npy'

# test for npy file before loading - run creation script, if data is absent
if not os.path.isfile(mnist_fname):
    # download and save data from the web
    from MNIST_all import MNIST_makeAll
    MNIST_makeAll.download_save()

# load mnist
mnist = np.load(mnist_fname, allow_pickle = True).item()

class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
max_ind = 999
crop = 2
downsample_ratio = 2
downsample_method = 1

## test generate_ds_mnist
# generate_ds_mnist( max_ind, class_labels, crop, downsample_ratio, downsample_method,
# inds_to_ave, pixel_sum, inds_to_calc_RF, num_features, screen_size, save_results_folder,
# show_thumbnails )
generate_ds_mnist(
                  max_ind,
                  class_labels,
                  crop,
                  downsample_ratio,
                  downsample_method,
                  [i for i in range(550,1000)],
                  6,
                  [i for i in range(550,1000)],
                  85, (1920, 1080), '', 0
                 )

## test extract_mnist_feature_array
# extract_mnist_feature_array( mnist, labels, image_indices, phase_label )
test_image_array = extract_mnist_feature_array(mnist, class_labels, range(max_ind+1), 'train')

# test crop_downsample_vectorize_images
# crop_downsample_vectorize_images( im_stack, crop_val, downsample_ratio, downsample_method )
crop_downsample_vectorize_images(test_image_array[...,0], crop, downsample_ratio, downsample_method)

# test average_image_stack
# average_image_stack( im_stack, indices_to_average )

# test select_active_pixels
# select_active_pixels( feature_array, num_features, screen_size,
#    save_image_folder=[], show_thumbnails=0 )
