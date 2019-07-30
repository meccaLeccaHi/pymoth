#!/usr/bin/env python3

def download_save():
	'''
	saves the following dictionary, called mnist, to .npy file
    train_images: np.array[28x28x60000]
    test_images: np.array[28x28x10000]
    train_labels: np.array[60000x1]
    test_labels: np.array[10000x1]

	Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
	MIT License
	'''

	import os
	import numpy as np
	# from MNIST_all import MNIST_read

	# # download and save data from Yann Lecun's website
	# [train_imgs, train_lbls, test_imgs, test_lbls] = MNIST_read.read();

	# download and save data from Keras
	from keras.datasets import mnist

	# directory to save image data
	im_dir = 'MNIST_all'

	(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()

	mnist = {
				'train_images':train_imgs,
				'test_images':test_imgs,
				'train_labels':train_lbls,
				'test_labels':test_lbls,
			}

	np.save(os.path.dirname(__file__) + os.sep + im_dir + os.sep + 'MNIST_all.npy', mnist)

if __name__ == "__main__":
    main()

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
