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

def cropDownsampleVectorizeImageStack( imStack, cropVal, dsVal, downsampleMethod ):
    '''
    For each image in a stack of images: Crop, then downsample, then make into a col vector.
    Inputs:
        1. imStack = numImages x width x height array
        2. cropVal = number of pixels to shave off each side. can be a scalar or a
            4 x 1 vector: top, bottom, left, right.
        3. dsVal = amount to downsample
        4. downsampleMethod: if 0, do downsampling by summing square patches.
            If 1, use bicubic interpolation.
    Output:
        1. imArray = a x numImages array, where a = number of pixels in the cropped and downsampled images

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    from scipy.misc import imresize
    import numpy as np

    if type(cropVal) is int:
        cropVal = cropVal*np.ones(4,dtype = int)

    if len(imStack.shape)==3:
        im_z,im_height,im_width = imStack.shape
    else:
        im_height,im_width = imStack.shape
        im_z = 1

    width = range(cropVal[2], im_width-cropVal[3])
    height = range(cropVal[0], im_height-cropVal[1])

    new_width = (im_width-np.sum(cropVal[2:]))/dsVal
    new_height = (im_height-np.sum(cropVal[0:2]))/dsVal

    imColArray = np.zeros((int(new_width*new_height),im_z))
    # crop, downsample, vectorize the thumbnails one-by-one
    for s in range(im_z):
        t = imStack[s,...]
        # crop image
        ixgrid = np.ix_(width, height)
        t = t[ixgrid]

        if downsampleMethod: # bicubic
            t2 = imresize(t, 1/dsVal, interp='bicubic')

        else: # sum 2 x 2 blocks
            t2 = np.zeros((int(len(height)/dsVal),int(len(width)/dsVal)))
            for i in range(int(len(height)/dsVal)):
                for j in range(int(len(width)/dsVal)):
                    b = t[(i-1)*dsVal+1:i*dsVal+1, (j-1)*dsVal+1:j*dsVal+1]
                    t2[i,j] = b.sum()

        imColArray[:,s] = t2.flatten()/t2.max()

    return imColArray

def averageImageStack( imStack, indicesToAverage ):
    '''
    Average a stack of images
    Inputs:
        1. imStack = 3-d stack (x, y, z) OR 2-d matrix (images-as-col-vecs, z)
        Caution: Do not feed in featureArray (ie 3-d with dim 1 = feature cols, 2 = samples per class, 3 = classes)
        2. indicesToAverage: which images in the stack to average
    Output:
        1. averageImage: (if input is 3-d) or column vector (if input is 2-d)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    import numpy as np

    imStack_shape = imStack.shape

    # case: images are col vectors
    if len(imStack_shape) == 2:
        aveIm = np.zeros((imStack_shape[0],))
    else:
        aveIm = np.zeros(imStack_shape)

    for i in indicesToAverage:
        aveIm += imStack[:, i]

    # normalize
    aveIm /= imStack_shape[1]

    return aveIm

def selectActivePixels( featureArray, numFeatures, saveImageFolder=[],
    scrsz = (1920, 1080), showThumbnails = 0 ):
    '''
    Select the most active pixels, considering all class average images, to use as features.
    Inputs:
        1. featureArray: 3-D array nF x nS x nC, where nF = # of features,
        nS = # samples per class, nC = number of classes. As created by genDS_MNIST.
        2. numFeatures: The number of active pixels to use (these form the receptive field).
        3. saveImageFolder: dir to save average class images, empty = don't save
        4. screensize: (width, height)
        5. showThumbnails: number of thumbnails to plot
    Output:
        1. activePixelInds: 1 x nF vector of indices to use as features.
        Indices are relative to the vectorized thumbnails (so between 1 and 144).

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    # make a classAves matrix (cA), each col a class ave 1 to 10 (ie 0),
    #  and add a col for the overallAve
    import numpy as np
    from support_functions.extractFA import averageImageStack
    from support_functions.show_figs import showFeatureArrayThumbnails

    pixNum, numPerClass, classNum  = featureArray.shape
    cA = np.zeros((pixNum, classNum+1))

    for i in range(classNum):

        cA[:,i] = averageImageStack(featureArray[:,:,i], list(range(numPerClass)))

    # last col = average image over all digits
    cA[:,-1] = np.sum(cA[:,:-1], axis=1) / classNum

    # normed version (does not rescale the overall average)
    z = np.max(cA, axis=0)
    z[-1] = 1
    caNormed = cA/np.tile(z, (pixNum,1))
    # num = size(caNormed,2);

    # select most active 'numFeatures' pixels
    this = cA[:, :-1]

    thisLogical = np.zeros(this.shape)

    # all the pixel values from all the class averages, in descending order
    vals = np.sort(this.flatten())[::-1]

    # start selecting the highest-valued pixels
    stop = 0
    while not stop:
        thresh = vals.max()
        thisLogical[this>=thresh] = 1
        activePixels = thisLogical.sum(axis=1) # sum the rows
        # If a class ave had the i'th pixel, selected, keptPixels(i) > 0
        stop = (activePixels > 0).sum() >= numFeatures # check if we have enough pixels

        vals = vals[vals < thresh]  # peel off the value(s) just used

    activePixelInds = np.nonzero(activePixels > 0)[0]

    if showThumbnails and saveImageFolder:
        # plot the normalized classAves pre-ablation
        normalize = 0
        titleStr = 'class aves, all pixels'
        showFeatureArrayThumbnails(caNormed, classNum+1, normalize, titleStr,
            scrsz, saveImageFolder, 'all')

        # look at active pixels of the classAves, ie post-ablation
        normalize = 0
        caActiveOnly = np.zeros(caNormed.shape)
        caActiveOnly[activePixelInds, : ] = caNormed[activePixelInds, :]
        titleStr = 'class aves, active pixels only'

        showFeatureArrayThumbnails(caActiveOnly, classNum+1, normalize, titleStr,
            scrsz, saveImageFolder, 'active')

    return activePixelInds

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
