def MNIST_read():

	# Read in MNIST digit set in Le Cun's format
	# [trainImages, trainLabels, testImages, testLabels] = MNIST_read()
	#
	# The data is available at
	# http://yann.lecun.com/exdb/mnist/ 
	# 
	# OUTPUT:
	# trainImages(:,:,i) is a numpy matrix of size 28x28x60000
	#		 0 = background, 255 = foreground
	# trainLabels(i) - 60000x1 vector of integer
	# testImages(:,:,i) size 28x28x10,000
	# testLabels(i)
	#
	# Use MNIST_show(trainImages, trainLabels) to visualize data.
	
	import numpy as np
	import os

	im_dir = 'MNIST_all'
	raw_dir = 'raw'

	# Check if all processed files exist
	fileTest = all([not os.path.isfile(os.path.join(im_dir,fn)) for fn in ['train.npy','test.npy']])

	if fileTest:
		np.load('train.npy')
	else:
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

		# Check for raw (compressed) image files		
		for set_label, urls in image_urls.items():
			# Check for images
			if not os.path.isfile(os.path.join('.',im_dir,raw_dir,set_label+'_images.gz')):
				print(f'Downloading {set_label} images')
				wget.download(urls['images'],(os.path.join('.',im_dir,raw_dir,set_label+'_images.gz')))
			# Check for labels
			if not os.path.isfile(os.path.join('.',im_dir,raw_dir,set_label+'_labels.gz')):
				print(f'Downloading {set_label} labels')
				wget.download(urls['labels'],(os.path.join('.',im_dir,raw_dir,set_label+'_labels.gz')))

			# Create processed (numpy) image files
			def read_img(filename):
				with gzip.open(filename) as f:
					zero, data_type, dims = struct.unpack('>HBB', f.read(4))
					shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
					return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
			# See reference here: gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40

#			def read_lbl(filename):
#				with gzip.open(filename) as f:
#					zero, data_type, dims = struct.unpack('>HBB', f.read(4))
#					shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
#					return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

			print(f'Reading {set_label} images')
			image_data = read_img(os.path.join('.',im_dir,raw_dir,set_label+'_images.gz'))
			print(f'Saving {set_label} images')
			np.save(os.path.join('.',im_dir,set_label+'.npy'),image_data)

	#return images (after they've been loaded)

		
	#fid = fopen('train-images-idx3-ubyte','r','ieee-be');  big endian
	#A = fread(fid,4,'uint32');
	#num_images = A(2);
	#mdim = A(3);
	#ndim = A(4);

	#train_images = fread(fid,mdim*ndim*num_images,'uint8=>uint8');
	#train_images = reshape(train_images,[mdim, ndim,num_images]);
	#train_images = permute(train_images, [2 1 3]); 

	#fclose(fid);


	#fid = fopen('train-labels-idx1-ubyte','r','ieee-be');
	#A = fread(fid,2,'uint32');
	#num_images = A(2);

	#train_labels = fread(fid,num_images,'uint8=>uint8');

	#fclose(fid);


	# Test

	#fid = fopen('t10k-images-idx3-ubyte','r','ieee-be');
	#A = fread(fid,4,'uint32');
	#num_images = A(2);
	#mdim = A(3);
	#ndim = A(4);

	#test_images = fread(fid,mdim*ndim*num_images,'uint8=>uint8');
	#test_images = reshape(test_images,[mdim, ndim,num_images]);
	#test_images = permute(test_images, [2 1 3]); 

	#fclose(fid);

	# Testing labels:
	#fid = fopen('t10k-labels-idx1-ubyte','r','ieee-be');
	#A = fread(fid,2,'uint32');
	#num_images = A(2);

	#test_labels = fread(fid,num_images,'uint8=>uint8');

	#fclose(fid);



	# return [train_images, train_labels, test_images, test_labels]

