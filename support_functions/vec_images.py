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
