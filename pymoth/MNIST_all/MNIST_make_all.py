#!/usr/bin/env python3

import numpy as _np
from keras.datasets import mnist as _mnist # also requires tensorflow

def make_MNIST(mnist_fpath):
	'''
	Save the following data to .npy file:
		train_images: np.array[28x28x60000]
		test_images: np.array[28x28x10000]
		train_labels: np.array[60000x1]
		test_labels: np.array[10000x1]

	Args:
		mnist_fpath (str): Path and filename for data to be saved under in the \
		user's Home (~) directory.
	'''

	# from MNIST_all import MNIST_read

	# # download and save data from Yann Lecun's website
	# [train_imgs, train_lbls, test_imgs, test_lbls] = MNIST_read.read();

	## download and save data from Keras
	# directory to save image data
	# im_dir = 'MNIST_all'

	(train_imgs, train_lbls), (test_imgs, test_lbls) = _mnist.load_data()

	mnist = {
				'train_images':train_imgs,
				'test_images':test_imgs,
				'train_labels':train_lbls,
				'test_labels':test_lbls,
			}

	_np.save(mnist_fpath, mnist)
	print('MNIST data saved:', mnist_fpath)

if __name__ == "__main__":
    make_MNIST()

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
