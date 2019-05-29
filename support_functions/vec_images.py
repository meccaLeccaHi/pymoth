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
