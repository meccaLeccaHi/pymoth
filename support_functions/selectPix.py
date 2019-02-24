def selectActivePixels( featureArray, numFeatures, showImages ):
    # Select the most active pixels, considering all class average images, to use as features.
    # Inputs:
    #    1. featureArray: 3-D array nF x nS x nC, where nF = # of features, nS = # samples per class, nC = number of classes.
    #        As created by generateDwnsampledMnistSet_fn.m
    #    2. numFeatures: The number of active pixels to use (these form the receptive field).
    #    3. showImages:  1 means show average class images, 0 = don't show.
    # Output:
    #   1. activePixelInds: 1 x nF vector of indices to use as features.
    #       Indices are relative to the vectorized thumbnails (so between 1 and 144).

    # make a classAves matrix, each col a class ave 1 to 10 (ie 0), and add a col for the overallAve

    import numpy as np
    from support_functions.aveImStack import averageImageStack

    pixNum, numPerClass, classNum  = featureArray.shape
    cA = np.zeros((pixNum, classNum+1))

    print(pixNum, numPerClass, classNum)

    for i in range(classNum):
        #temp = np.zeros((pixNum, numPerClass))
        foo = averageImageStack(featureArray[:,:,i], list(range(numPerClass)))
        # print(foo.shape)
        # print(cA[:,i].shape)
        cA[:,i] = averageImageStack(featureArray[:,:,i], list(range(numPerClass)))

    # # make a classAves matrix, each col a class ave 1 to 10 (ie 0), and add a col for the overallAve
    # numPerClass = size(featureArray,2);
    # cA = zeros(size(featureArray,1), size(featureArray,3) + 1);
    #
    # for i = 1:size(featureArray,3)
    #     # change dim of argin 1 to 'averageImageStack'
    #     temp = zeros(size(featureArray,1), size(featureArray,2));
    #     temp(:,:) = featureArray(:,:,i);
    #     cA(:,i) = averageImageStack_fn(temp, 1:numPerClass );
    # end
    # # last col = overall average image:
    # cA(:,end) = sum( cA(:,1:end-1), 2) / (size(cA,2) - 1) ;
    #
    # # normed version. Do not rescale the overall average:
    # z = max(cA);
    # caNormed = cA./repmat( [z(1:end-1), 1], [size(cA,1),1]);
    # num = size(caNormed,2);
    #
    # # select most active 'numFeatures' pixels:
    # this = cA( : , 1:end - 1 );
    # thisLogical = zeros( size( this ) );
    # vals = sort( this(:), 'descend' );    # all the pixel values from all the class averages, in descending order
    # # start selecting the highest-valued pixels:
    # stop = 0;
    # while ~stop
    #     thresh = max(vals);
    #     thisLogical( this >= thresh ) = 1;
    #     activePixels = sum( thisLogical,  2 );  # sum the rows. If a class ave had the i'th pixel, selected, keptPixels(i) > 0
    #     stop = sum(activePixels > 0) >= numFeatures;  # we have enough pixels.
    #     vals = vals(vals < thresh);  # peel off the value(s) just used.
    # end
    # activePixelInds = find( activePixels > 0 );
    #
    # if showImages
    #     # plot the normalized classAves pre-ablation:
    #     normalize = 0;
    #     titleStr = 'class aves, all pixels';
    #     showFeatureArrayThumbnails_fn(caNormed, size(caNormed,2), normalize, titleStr)
    #
    #     # look at active pixels of the classAves, ie post-ablation:
    #     normalize = 0;
    #     caActiveOnly = zeros(size(caNormed));
    #     caActiveOnly(activePixelInds, : ) = caNormed(activePixelInds, :) ;
    #     titleStr = 'class aves, active pixels only';
    #     showFeatureArrayThumbnails_fn(caActiveOnly, size(caActiveOnly,2), normalize, titleStr)
    #
    # end


    return activePixelInds
