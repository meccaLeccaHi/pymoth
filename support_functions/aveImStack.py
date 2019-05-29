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
