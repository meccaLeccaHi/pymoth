import gzip
import numpy as np
import struct

filename = './MNIST_all/raw/train_images.gz'
fname_lbl = './MNIST_all/raw/train_labels.gz'

with gzip.open(filename) as f:
	zero, data_type, dims = struct.unpack('>HBB', f.read(4))
	shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
	#print('shape=',shape)
	print(len(np.fromstring(f.read(), dtype=np.uint8).reshape(shape)))


with open(fname_lbl, 'rb') as flbl:
	magic, num = struct.unpack(">II", flbl.read(8))
	lbl = np.fromfile(flbl, dtype=np.int8)
	print(len(lbl))
