def showFeatureArrayThumbnails( featureArray, numPerClass, normalize, titleString ):
    # Show thumbnails of inputs used in the experiment.
    # Inputs:
    #   1. featureArray = either 3-D (1 = cols of features, 2 = within class samples, 3 = class)
    #                               or 2-D (1 = cols of features, 2 = within class samples, no 3)
    #   2. numPerClass = how many of the thumbnails from each class to show.
    #   3. normalize = 1 if you want to rescale thumbs to [0 1], 0 if you don't
    #   4. titleString = string

    # tkinter errors if run after matplotlib is loaded, so we run it first
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    scrsz = [screen_width, screen_height]

    import numpy as np
    import matplotlib.pyplot as plt

    # bookkeeping: change dim if needed
    # DEV NOTE: Clarify with Charles - this seems unnecessary
    if len(featureArray.shape)==2:
        f = np.zeros((featureArray.shape[0],featureArray.shape[1],1))
        f[:,:,0] = featureArray
        featureArray = f  #.squeeze()

    print('featureArray shape:',featureArray.shape)

    pixNum, numPerClass, nC  = featureArray.shape
    # DEV NOTE: Should be able to remove classNum from inputs above

    total = nC*numPerClass # total number of subplots
    numRows = np.ceil(np.sqrt(total/2)) # n of rows
    numCols = np.ceil(np.sqrt(total*2)) # n of cols
    vert = 1/(numRows + 1) # vertical step size
    horiz = 1/(numCols + 1) # horizontal step size

    fig_sz = [np.floor((i/100)*0.8) for i in scrsz]
    print('fig_sz',fig_sz)
    print(numRows,numCols)
    thumbs = plt.figure(figsize=fig_sz, dpi=100)

    for cl in range(nC): # 'class' is a keyword in Python; renamed to 'cl'
        for i in range(numPerClass):
            ax_i = numPerClass*(cl) + i + 1
            thisInput = featureArray[:, i, cl]

            if normalize:
                # DEV NOTE: This only affects the last image in the stack (the average)
                # -> could be made more efficient
                thisInput /= thisInput.max() # renormalize, to offset effect of classMagMatrix scaling

            ax_count = i + (cl*nC)
            plt.subplot(np.int(numRows),np.int(numCols),ax_i)

            # # DEV NOTE: Rename or delete these
            # # reverse:
            # # thisInput = (-thisInput + 1)*1.1
            # thisCol = ax_i % numCols
            # if thisCol==0:
            #     thisCol = numCols
            # thisRow = np.ceil( ax_i / numCols )

            # a = horiz*(thisCol - 1) # x-coordinates (left edge)
            # b = 1 - vert*(thisRow) # y-coordinates (bottom edge)
            # c = horiz # subplot width
            # d = vert # subplot height

            side = np.int(np.sqrt(len(thisInput)))
            plt.imshow(thisInput.reshape((side,side)), cmap='gray')

    # add a title at the bottom
    plt.xlabel(titleString, fontweight='bold')
    plt.show()
