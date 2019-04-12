def selectActivePixels( featureArray, numFeatures, showImages ):
    '''
    Select the most active pixels, considering all class average images, to use as features.
    Inputs:
        1. featureArray: 3-D array nF x nS x nC, where nF = # of features,
        nS = # samples per class, nC = number of classes. As created by genDS_MNIST.
        2. numFeatures: The number of active pixels to use (these form the receptive field).
        3. showImages:  1 means show average class images, 0 = don't show.
    Output:
        1. activePixelInds: 1 x nF vector of indices to use as features.
        Indices are relative to the vectorized thumbnails (so between 1 and 144).
    '''

    # make a classAves matrix, each col a class ave 1 to 10 (ie 0), and add a col for the overallAve
    import numpy as np
    from support_functions.aveImStack import averageImageStack
    from support_functions.show_figs import showFeatureArrayThumbnails

    pixNum, numPerClass, classNum  = featureArray.shape
    cA = np.zeros((pixNum, classNum+1))

    for i in range(classNum):
        #temp = np.zeros((pixNum, numPerClass))
        cA[:,i] = averageImageStack(featureArray[:,:,i], list(range(numPerClass)))

    # last col = average image over all digits
    cA[:,-1] = np.sum(cA[:,:-1],axis=1) / classNum

    # normed version (does not rescale the overall average)
    z = np.max(cA,axis=0)
    z[-1] = 1
    caNormed = cA/np.tile(z,(pixNum,1))
    # num = size(caNormed,2);

    # select most active 'numFeatures' pixels
    this = cA[:,:-1]
    thisLogical = np.zeros((pixNum, classNum))

    # all the pixel values from all the class averages, in descending order
    vals = this.flatten()
    vals.sort()
    vals = vals[::-1]

    # start selecting the highest-valued pixels
    # DEV NOTE: Clarify this part with CBD - (why do it?)
    stop = 0
    while not stop:
        thresh = vals.max()
        thisLogical[this>=thresh] = 1
        activePixels = thisLogical.sum(axis=1) # sum the rows.
        # If a class ave had the i'th pixel, selected, keptPixels(i) > 0
        stop = (activePixels > 0).sum() >= numFeatures # check if we have enough pixels

        vals = vals[vals < thresh]  # peel off the value(s) just used

    activePixelInds = np.nonzero(activePixels > 0)[0]
    print('activePixelInds len:', len(activePixelInds))
    print('activePixelInds[:5]:', activePixelInds[:5])
    print("NEED TO FIX?: Doesn't correspond to matlab counterpart")
    # quit()

    if showImages:
        # plot the normalized classAves pre-ablation
        normalize = 0
        titleStr = 'class aves, all pixels'
        showFeatureArrayThumbnails(caNormed, classNum+1, normalize, titleStr)

        # look at active pixels of the classAves, ie post-ablation
        normalize = 0
        caActiveOnly = np.zeros(caNormed.shape)
        caActiveOnly[activePixelInds, : ] = caNormed[activePixelInds, :]
        titleStr = 'class aves, active pixels only'

        showFeatureArrayThumbnails(caActiveOnly, classNum+1, normalize, titleStr)

    return activePixelInds
