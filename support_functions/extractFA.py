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

    import numpy as np

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
