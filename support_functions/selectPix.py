def selectActivePixels( featureArray, numFeatures, saveImageFolder=[], scrsz = (1920, 1080) ):
    '''
    Select the most active pixels, considering all class average images, to use as features.
    Inputs:
        1. featureArray: 3-D array nF x nS x nC, where nF = # of features,
        nS = # samples per class, nC = number of classes. As created by genDS_MNIST.
        2. numFeatures: The number of active pixels to use (these form the receptive field).
        3. saveImageFolder:  dir to save average class images, empty = don't save
    Output:
        1. activePixelInds: 1 x nF vector of indices to use as features.
        Indices are relative to the vectorized thumbnails (so between 1 and 144).
    '''

    # make a classAves matrix (cA), each col a class ave 1 to 10 (ie 0),
    #  and add a col for the overallAve
    import numpy as np
    from support_functions.aveImStack import averageImageStack
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
    # DEV NOTE: Clarify this part with CBD - (why do it?)
    stop = 0
    while not stop:
        thresh = vals.max()
        thisLogical[this>=thresh] = 1
        activePixels = thisLogical.sum(axis=1) # sum the rows
        # If a class ave had the i'th pixel, selected, keptPixels(i) > 0
        stop = (activePixels > 0).sum() >= numFeatures # check if we have enough pixels

        vals = vals[vals < thresh]  # peel off the value(s) just used

    activePixelInds = np.nonzero(activePixels > 0)[0]
    print(f"FOLLOW-UP[{__file__}]")
    print("activePixelInds len:", len(activePixelInds))
    print('activePixelInds[:5]:', activePixelInds[:5])
    # NEED TO FIX?: Doesn't correspond to matlab counterpart
    # Same values for "thresh" for both versions
    # Same shape for "activePixelInds", but different values
    # import pdb; pdb.set_trace()

    if saveImageFolder:
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
