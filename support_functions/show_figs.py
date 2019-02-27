def showFeatureArrayThumbnails( featureArray, numPerClass, normalize, titleString ):
    # Show thumbnails of inputs used in the experiment.
    # Inputs:
    #   1. featureArray = either 3-D (1 = cols of features, 2 = within class samples, 3 = class)
    #                               or 2-D (1 = cols of features, 2 = within class samples, no 3)
    #   2. numPerClass = how many of the thumbnails from each class to show.
    #   3. normalize = 1 if you want to rescale thumbs to [0 1], 0 if you don't
    #   4. titleString = string

    import numpy as np

    # bookkeeping: change dim if needed
    # DEV NOTE: Clarify with Charles
    print('larg',featureArray.shape)
    if len(featureArray.shape)==2:
        print('WARNING: Utilizing un-tested feature!')
        f = np.zeros((featureArray.shape[0],featureArray.shape[1],1))
        f[:,:,0] = featureArray
        featureArray = f

    pixNum, nC, classNum  = featureArray.shape
    # DEV NOTE: Should be able to remove classNum from inputs above and in script
    print('lalala- FIX THIS WHEN WE CAN COMPARE WITH THE MATLAB VARS')
    ## I think it's supposed to be 10x10, but need to check the vals of the matlab vars
    print(nC,numPerClass)
    total = nC*numPerClass
    # DEV NOTE: Add commentary when we understand better
    numRows = np.ceil(np.sqrt(total/2)) # param to set
    numCols = np.ceil(np.sqrt(total*2)) # param to set
    vert = 1/(numRows + 1)
    horiz = 1/(numCols + 1)
    #
    # scrsz = get(0,'ScreenSize');
    # thumbs = figure('Position',[scrsz(1), scrsz(2), scrsz(3)*0.8, scrsz(4)*0.8 ]);
    # for class = 1:nC
    #     for i = 1:numPerClass
    #         col = numPerClass*(class-1) + i;
    #         thisInput = featureArray(:, i, class) ;
    #         % show the thumbnail of the input:
    #         if normalize
    #             thisInput = thisInput/max(thisInput);  % renormalize, to offset effect of classMagMatrix scaling
    #         end
    # %        % reverse:
    # %        thisInput = (-thisInput + 1)*1.1;
    #         thisCol = mod( col, numCols );
    #         if thisCol == 0, thisCol = numCols; end
    #         thisRow = ceil( col / numCols );
    #         a = horiz*(thisCol - 1);
    #         b = 1 - vert*(thisRow);
    #         c = horiz;
    #         d = vert;
    #         subplot('Position', [a b c d] ), % [ left corner, bottom corner, width, height ]
    #         imshow(reshape(thisInput,[sqrt(length(thisInput)), sqrt(length(thisInput))] ) );   % Assumes square thumbnails
    #     end
    #    drawnow
    # end
    # % add a title at the bottom
    # xlabel(titleString, 'fontweight', 'bold' )
    # drawnow
