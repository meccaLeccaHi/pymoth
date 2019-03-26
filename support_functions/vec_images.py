def cropDownsampleVectorizeImageStack( imStack, cropVal, downsampleVal, downsampleMethod ):
    # For each image in a stack of images: Crop, then downsample, then make into a col vector.
    # Inputs:
    #   1. imStack = numImages x width x height array
    #   2. cropVal = number of pixels to shave off each side. can be a scalar or a
    #       4 x 1 vector: top, bottom, left, right.
    #   3. downsampleVal = amount to downsample
    #   4. downsampleMethod: if 0, do downsampling by summing square patches. If 1, use bicubic interpolation.
    # Output:
    #   1. imArray = a x numImages array, where a = number of pixels in the cropped and downsampled images

    from scipy.misc import imresize
    import numpy as np

    if type(cropVal) is int:
        cropVal = cropVal*np.ones((1,4),dtype = int)[0]

    if len(imStack.shape)==3:
        z,h,w = imStack.shape
    else:
        h,w = imStack.shape
        z = 1

    width = range(cropVal[2], w-cropVal[3])
    height = range(cropVal[0], h-cropVal[1])

    new_width = (h-np.sum(cropVal[0:2]))/downsampleVal
    new_height = (w-np.sum(cropVal[2:]))/downsampleVal

    imColArray = np.zeros((int(new_width*new_height),z))
    d = downsampleVal
    # crop, downsample, vectorize the thumbnails one-by-one
    for s in range(z):
        t = imStack[s,...]
        # crop image
        ixgrid = np.ix_(width, height)
        t = t[ixgrid]

        if downsampleMethod: # bicubic
            t2 = imresize(t,1/downsampleVal, interp='bicubic')

        else: # sum 2 x 2 blocks
            t2 = np.zeros((int(len(height)/d),int(len(width)/d)))
            for i in range(int(len(height)/d)):
                for j in range(int(len(width)/d)):
                    b = t[(i-1)*d+1:i*d+1, (j-1)*d+1:j*d+1]
                    t2[i,j] = b.sum()

        t2 = t2.flatten()
        t2 = t2/np.max(t2)

        imColArray[:,s] = t2

    return imColArray
