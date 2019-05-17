def downloadAndSave():
	'''
	saves the following dictionary, called mnist, to .npy file
    train_images: np.array[28x28x60000]
    test_images: np.array[28x28x10000]
    train_labels: np.array[60000x1]
    test_labels: np.array[10000x1]
	'''

	import os
	import numpy as np
	from MNIST_all import MNIST_read

	[train_imgs, train_lbls, test_imgs, test_lbls] = MNIST_read.read();

	im_dir = 'MNIST_all'

	mnist = {
				'train_images':train_imgs,
				'test_images':test_imgs,
				'train_labels':train_lbls,
				'test_labels':test_lbls,
			}

	np.save(os.path.join('.',im_dir,'MNIST_all.npy'), mnist)

if __name__ == "__main__":
    main()
