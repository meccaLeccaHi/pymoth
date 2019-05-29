def read():
	'''
	Read in MNIST digit set in Le Cun's format
	[trainImages, trainLabels, testImages, testLabels] = MNIST_read()

	The data is available at:
	http://yann.lecun.com/exdb/mnist/

	OUTPUT:
	trainImages is a numpy matrix of size 60,000x28x28
			 (0 = background, 255 = foreground)
	trainLabels is a 60,000x1 vector of integers
	testImages is a numpy matrix of size 10,000x28x28
	testLabels is a 10,000x1 vector of integers

	Use MNIST_show(trainImages, trainLabels) to visualize data.

	Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
	MIT License
	'''

	import numpy as np
	import os

	im_dir = 'MNIST_all'
	raw_dir = 'raw'
	file_list = ['train_images.npy','train_labels.npy','test_images.npy','test_labels.npy']
	# Check if all processed files exist
	missingFileTest = any(
			[not os.path.isfile(os.path.join('.',im_dir,raw_dir,fn)) for fn in file_list])

	if missingFileTest:

		print('Downloading data')

		import wget
		import gzip
		import struct

		image_urls = {
			'train':{
				'images':'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
				'labels':'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'},
			'test':{
				'images':'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
				'labels':'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}
		}

		# Define function to process (via numpy) image files
		def extract_images(gz_file):
			with gzip.open(os.path.join('.',im_dir,raw_dir,gz_file)) as f:
				zero, data_type, dims = struct.unpack('>HBB', f.read(4))
				shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
				images = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
			print(f'Image array shape: {shape}')
			return images
		# Define function to process (via numpy) image label files
		def extract_labels(gz_file):
			with gzip.open(os.path.join('.',im_dir,raw_dir,gz_file)) as f:
				f.read(8)
				labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
			print(f'Image label array shape: {labels.shape}')
			return labels

		# Loop each exp. set [training and test]
		for set_label, urls in image_urls.items():
			# Check for /raw folder, create one if doesn't exist
			if not os.path.isdir(os.path.join('.',im_dir,raw_dir)):
				os.mkdir(os.path.join('.',im_dir,raw_dir))
			# Check for raw (compressed) images
			if not os.path.isfile(os.path.join('.',im_dir,raw_dir,set_label+'_images.gz')):
				print(f'Downloading {set_label} images')
				wget.download(urls['images'],os.path.join('.',im_dir,raw_dir,set_label+'_images.gz'))
			# Check for raw (compressed) labels
			if not os.path.isfile(os.path.join('.',im_dir,raw_dir,set_label+'_labels.gz')):
				print(f'Downloading {set_label} labels')
				wget.download(urls['labels'],os.path.join('.',im_dir,raw_dir,set_label+'_labels.gz'))

			print(f'Reading {set_label} images')
			images = extract_images(set_label+'_images.gz')
			print(f'Reading {set_label} labels')
			labels = extract_labels(set_label+'_labels.gz')
			print(f'Saving {set_label} images and labels')
			np.save(os.path.join('.',im_dir,raw_dir,set_label+'_images.npy'),images)
			np.save(os.path.join('.',im_dir,raw_dir,set_label+'_labels.npy'),labels)

	train_imgs = np.load(os.path.join('.',im_dir,raw_dir,'train_images.npy'))
	train_lbls = np.load(os.path.join('.',im_dir,raw_dir,'train_labels.npy'))
	test_imgs = np.load(os.path.join('.',im_dir,raw_dir,'test_images.npy'))
	test_lbls = np.load(os.path.join('.',im_dir,raw_dir,'test_labels.npy'))

	return train_imgs, train_lbls, test_imgs, test_lbls

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
