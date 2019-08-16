#!/usr/bin/env python3
import os
import numpy as np

# import packages and modules
from .generate import generate_ds_mnist, extract_mnist_feature_array, \
    crop_downsample_vectorize_images, average_image_stack, select_active_pixels

def main():

    print('Testing generate module:')

    # generate dummy data
    mnist_fname = '/tmp/foo.npy'

    class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    max_ind = 999
    crop = 2
    downsample_ratio = 2
    downsample_method = 1
    screen_size = (1920, 1080)

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
                      85,
                      screen_size, '',
                      0,
                     )
    print('\tgenerate_ds_mnist function test passed')

    # load mnist
    mnist = np.load(mnist_fname, allow_pickle = True).item()

    # remove temporary data files
    os.remove(mnist_fname)

    ## test extract_mnist_feature_array
    # extract_mnist_feature_array( mnist, labels, image_indices, phase_label )
    dummy_image_array = extract_mnist_feature_array(
                    mnist,
                    class_labels,
                    range(max_ind+1),
                    'train'
                    )
    print('\textract_mnist_feature_array function test passed')

    # test crop_downsample_vectorize_images
    # crop_downsample_vectorize_images( im_stack, crop_val, downsample_ratio, downsample_method )
    crop_downsample_vectorize_images(
                    dummy_image_array[...,0],
                    crop,
                    downsample_ratio,
                    downsample_method
                    )
    print('\tcrop_downsample_vectorize_images function test passed')


    im_z, im_height, im_width, label_len = dummy_image_array.shape
    dummy_feature_array = np.ones((144, im_z, label_len))

    # test average_image_stack
    # average_image_stack(im_stack, indices_to_average)
    average_image_stack(dummy_feature_array[...,0], list(range(5)))
    print('\taverage_image_stack function test passed')

    # test select_active_pixels
    # select_active_pixels( feature_array, num_features, screen_size,
    #    save_image_folder=[], show_thumbnails=0 )
    select_active_pixels(dummy_feature_array, 85, screen_size)
    print('\tselect_active_pixels function test passed')

if __name__ == '__main__':
    main()
