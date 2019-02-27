def averageImageStack( imStack, indicesToAverage ):
    # Average a stack of images:
    # inputs:
    # 1. imStack = 3-d stack (x, y, z) OR 2-d matrix (images-as-col-vecs, z)
    # Caution: Do not feed in featureArray (ie 3-d with dim 1 = feature cols, 2 = samples per class, 3 = classes)
    # 2. indicesToAverage: which images in the stack to average
    # Output:
    # 1. averageImage: (if input is 3-d) or column vector (if input is 2-d)

    import numpy as np

    imStack_shape = imStack.shape

    # case: images are col vectors
    if len(imStack_shape) == 2:
        aveIm = np.zeros((imStack_shape[0],))

        for i in indicesToAverage:
            aveIm += imStack[:, i]
    else:
        aveIm = np.zeros(imStack_shape)
        # DEV NOTE: Don't actually need for loop below in each branch of if construct
        # - fix during refactor
        for i in indicesToAverage:
            aveIm += imStack[:, i]

    # normalize
    averageIm = aveIm/imStack_shape[1]
    # DEV NOTE: remove averageIm above feed right side directly to return to save memory

    return averageIm
