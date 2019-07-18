import numpy as np
import os
import matplotlib.pyplot as plt

def show_FA_thumbs( feature_array, show_per_class, normalize, title_string,
    screen_size, images_filename ):
    '''
    Show thumbnails of inputs used in the experiment.
    Inputs:
        1. feature_array = either 3-D
            (1 = cols of features, 2 = within class samples, 3 = class)
            or 2-D (1 = cols of features, 2 = within class samples, no 3)
        2. show_per_class = how many of the thumbnails from each class to show.
        3. normalize = 1 if you want to rescale thumbs to [0 1], 0 if you don't
        4. title_string = string
        5. screen_size = tuple
        6. images_filename = string (including path)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    if images_filename:
        images_folder = os.path.dirname(images_filename)

    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
        print('Creating results directory: {}'.format(images_filename))

    # bookkeeping: change dim if needed
    if len(feature_array.shape)==2:
        f = np.zeros((feature_array.shape[0],feature_array.shape[1],1))
        f[:,:,0] = feature_array
        feature_array = f[:,:,:]  #.squeeze()

    pixNum, numPerClass, nC  = feature_array.shape

    total = nC*show_per_class # total number of subplots
    numRows = np.ceil(np.sqrt(total/2)) # n of rows
    numCols = np.ceil(np.sqrt(total*2)) # n of cols
    vert = 1/(numRows + 1) # vertical step size
    horiz = 1/(numCols + 1) # horizontal step size

    fig_sz = [np.floor((i/100)*0.5) for i in screen_size]
    thumbs_fig = plt.figure(figsize=fig_sz, dpi=100)

    for cl in range(nC): # 'class' is a keyword in Python; renamed to 'cl'
        for i in range(show_per_class):
            ax_i = show_per_class*(cl) + i + 1
            thisInput = feature_array[:, i, cl]

            # renormalize, to offset effect of classMagMatrix scaling
            if normalize:
                thisInput /= thisInput.max()

            ax_count = i + (cl*nC)
            plt.subplot(np.int(numRows),np.int(numCols),ax_i)

            side = np.int(np.sqrt(len(thisInput)))
            plt.imshow(thisInput.reshape((side,side)), cmap='gray', vmin=0, vmax=1)

    # add a title at the bottom
    plt.xlabel(title_string, fontweight='bold')

    # Save plot
    if os.path.isdir(images_folder):
        thumb_name = os.path.join(os.getcwd(), images_filename+'.png')
        thumbs_fig.savefig(thumb_name, dpi=100)
        print(f'Image thumbnails saved: {thumb_name}')

def show_EN_resp( simRes, modelParams, expP, show_acc_plots, show_time_plots,
                classLabels, screen_size, images_filename='' ):
    '''
    View readout neurons (EN):
        Color-code them dots by class and by concurrent octopamine.
        Collect stats: median, mean, and std of FR for each digit, pre- and post-training.
        Throughout, digits may be referred to as odors, or as odor puffs.
        'Pre' = naive. 'Post' = post-training

    Inputs:
        1. simRes: dictionary containing simulation results (output from sdeWrapper)
        2. modelParams: object containing model parameters for this moth
        3. expP: object containing experiment parameters with timing
            and digit class info from the experiment.
        4. show_acc_plots: show changes in accuracy.
        5. show_time_plots: show EN timecourses.
        6. classLabels: 1 to 10
        7. screen_size: tuple
        8. images_filename: to generate image filenames when saving (includes path).
            Optional argin. If this = '', images will not be saved (ie it's also a flag).

    Outputs (as fields of resultsStruct):
        1. preMeanResp = numENs x numOdors matrix = mean of EN pre-training
        2. preStdResp = numENs x numOdors matrix = std of EN responses pre-training
        3. ditto for post etc
        4. percentChangeInMeanResp = 1 x numOdors vector
        5. trained = list of indices corresponding to the odor(s) that were trained
        6. preSpontMean = mean(preSpont)
        7. preSpontStd = std(preSpont)
        8. postSpontMean = mean(postSpont)
        9. postSpontStd = std(postSpont)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    # from matplotlib import pyplot as plt
    if images_filename:
        images_folder = os.path.dirname(images_filename)

    # create directory for images (if doesnt exist)
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
        print('Creating results directory: {}'.format(images_filename))


    colors = [ (0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 0.3, 0.8), (0.8, 0.3, 1), (0.8, 1, 0.3), (0.5, 0.5, 0.5) ] # for 10 classes
    # concurrent octopamine will be marked with yellow x's

    if simRes['octo_hits'].max() > 0:
        octoTimes = simRes['T'][ simRes['octo_hits'] > 0 ]
    else:
        octoTimes = []

    # calc spont stats
    preSpont = simRes['E'][ np.logical_and(expP.preHebSpontStart < simRes['T'],
                                    simRes['T'] < expP.preHebSpontStop) ]
    postSpont = simRes['E'][ np.logical_and(expP.postHebSpontStart < simRes['T'],
                                    simRes['T'] < expP.postHebSpontStop) ]

    preHebMean = preSpont.mean()
    preHebStd = preSpont.std()
    postHebMean = postSpont.mean()
    postHebStd = postSpont.std()

    ## Set regions to examine:
    # 1. data from expP
    # simStart = expP.simStart
    # classMags = expP.classMags
    # stimStarts = expP.stimStarts # to get timeSteps from very start of sim
    stimStarts = expP.stimStarts*(expP.classMags > 0) # ie only use non-zero puffs
    # whichClass = expP.whichClass
    whichClass = expP.whichClass*(expP.classMags > 0)
    # startTrain = expP.startTrain
    # endTrain = expP.endTrain

    classList = np.unique(whichClass)
    # numClasses = len(classList)

    # pre-allocate list of empty dicts
    results = [dict() for i in range(modelParams.nE)]

    # Make one stats plot per EN. Loop through ENs:
    for enInd in range(modelParams.nE):

        thisEnResponse = simRes['E'][:, enInd]

        ## Calculate pre- and post-train odor response stats
        # Assumes that there is at least 1 sec on either side of an odor without octo

        # pre-allocate for loop
        preTrainOdorResp = np.full(len(stimStarts), np.nan)
        postTrainOdorResp = np.full(len(stimStarts), np.nan)

        for i, t in enumerate(stimStarts):
            # Note: to find no-octo stimStarts, there is a certain amount of machinery
            # in order to mesh with the timing data from the experiment.
            # For some reason octoTimes are not recorded exactly as listed in format
            # short mode. So we need to use abs difference > small thresh, rather
            # than ~ismember(t, octoTimes):
            small = 1e-8 # .00000001
            # assign no-octo, PRE-train response val (or -1)
            preTrainOdorResp[i] = -1 # as flag
            if (len(octoTimes)==0) or ((abs(octoTimes - t).min() > small) and (t < expP.startTrain)):
                resp_ind = np.logical_and(t-1 < simRes['T'], simRes['T'] < t+1)
                preTrainOdorResp[i] = thisEnResponse[resp_ind].max()

            # assign no-octo, POST-train response val (or -1)
            postTrainOdorResp[i] = -1
            if len(octoTimes)!=0:
                if (abs(octoTimes - t).min() > small) and (t > expP.endTrain):
                    resp_ind = np.logical_and(t-1 < simRes['T'], simRes['T'] < t+1)
                    postTrainOdorResp[i] = thisEnResponse[resp_ind].max()

        # pre-allocate for loop
        preMeanResp, preMedianResp, preStdResp, preNumPuffs, postMeanResp, \
            postMedianResp, postStdResp, postNumPuffs = \
            [np.full(len(classList), np.nan) for _ in range(8)]

        # calc no-octo stats for each odor, pre and post train:
        for k, cl in enumerate(classList):
            curCl = whichClass==cl
            preSA = preTrainOdorResp[np.logical_and(preTrainOdorResp>=0, curCl)]
            postSA = postTrainOdorResp[np.logical_and(postTrainOdorResp>=0, curCl)]

            ## calculate the averaged sniffs of each sample: SA means 'sniffsAveraged'
            # this will contain the average responses over all sniffs for each sample
            # DEV NOTE: Changed pretty drastically from orig version, but should be the same.
            # Double check with CBD
            if len(preSA)==0:
                preMeanResp[k] = -1
                preMedianResp[k] = -1
                preStdResp[k] = -1
                preNumPuffs[k] = 0
            else:
                preMeanResp[k] = preSA.mean()
                preMedianResp[k] = np.median(preSA)
                preStdResp[k] = preSA.std()
                preNumPuffs[k] = len(preSA)

            if len(postSA)==0:
                postMeanResp[k] = -1
                postMedianResp[k] = -1
                postStdResp[k] = -1
                postNumPuffs[k] = 0
            else:
                postMeanResp[k] = postSA.mean()
                postMedianResp[k] = np.median(postSA)
                postStdResp[k] = postSA.std()
                postNumPuffs[k] = len(postSA)

        # # to plot +/- 1 std of % change in meanResp, we want the std of our
        # # estimate of the mean = stdResp/sqrt(numPuffs). Make this calc:
        # preStdMeanEst = preStdResp/np.sqrt(preNumPuffs)
        # postStdMeanEst = postStdResp/np.sqrt(postNumPuffs)

        preSA = np.nonzero(preNumPuffs > 0)[0]
        postSA = np.nonzero(postNumPuffs > 0)[0]
        postOffset = postSA + 0.25

        # a key output:
        percentChangeInMeanResp = (100*(postMeanResp[preSA] - preMeanResp[preSA]))\
                                    /preMeanResp[preSA]
        percentChangeInNoiseSubtractedMeanResp = \
                                (100*(postMeanResp[preSA] - preMeanResp[preSA] - postHebMean))\
                                /preMeanResp[preSA]

        percentChangeInMedianResp = (100*(postMedianResp[preSA] - preMedianResp[preSA]))\
                                /preMedianResp[preSA]
        percentChangeInNoiseSubtractedMedianResp = \
                                (100*(postMedianResp[preSA] - preMedianResp[preSA] - postHebMean))\
                                /preMedianResp[preSA]

        # plot stats if wished:
        if show_acc_plots:
            fig_sz = [np.floor((i/100)*0.8) for i in screen_size]
            thisFig = plt.figure(figsize=fig_sz, dpi=100)

            # medians, pre and post
            ax = thisFig.add_subplot(2, 3, 1)
            ax.grid()
            ax.plot(preSA, preMedianResp[preSA], '*b')
            ax.plot(postOffset, postMedianResp[postSA], 'bo') # , markerfacecolor='b'
            #   ax.plot(pre, preMeanResp + preStdResp, '+g')
            #   ax.plot(post, postMeanResp + postStdResp, '+g')
            #   ax.plot(pre, preMeanResp - preStdResp, '+g')
            #   ax.plot(post, postMeanResp - postStdResp, '+g')

            # make the home EN of this plot red
            ax.plot(enInd, preMedianResp[enInd], 'ro')
            ax.plot(enInd + 0.25, postMedianResp[enInd], 'ro') # ,'markerfacecolor','r'
            ax.set_title(f'EN {enInd}\n median +/- std')
            ax.set_xlim([0, max(preSA) + 1])
            ax.set_ylim([0, 1.1*max(np.concatenate((preMedianResp, postMedianResp)))])
            ax.set_xticks(preSA, minor=False)
            ax.set_xticklabels(classLabels)

            # connect pre to post with lines for clarity
            for j in range(len(preSA)):
                if j==1:
                    lineColor = 'r'
                else:
                    lineColor = 'b'
                ax.plot(
                        [preSA[j], postOffset[j]],
                        [preMedianResp[preSA[j]], postMedianResp[preSA[j]]],
                        lineColor
                        )

            # percent change in medians
            ax = thisFig.add_subplot(2, 3, 2)
            ax.grid()
            ax.plot(
                preSA,
                (100*(postMedianResp[preSA] - preMedianResp[preSA]))/preMedianResp[preSA],
                'bo') # , markerfacecolor='b'
            # mark the trained odors in red:
            ax.plot(
                enInd,
                (100*(postMedianResp[enInd] - preMedianResp[enInd]))/preMedianResp[enInd],
                'ro') # , markerfacecolor='r'
            ax.set_title(r'% $\Delta$ median')
            ax.set_xlim([0, max(preSA)+1])
            # ax.set_ylim([-50,400])
            ax.set_xticks(preSA, minor=False)
            ax.set_xticklabels(classLabels)

            # relative changes in median, ie control/trained
            ax = thisFig.add_subplot(2, 3, 3)
            ax.grid()
            pn = np.sign(postMedianResp[enInd] - preMedianResp[enInd])
            y_vals = (pn * ( (postMedianResp[preSA] - preMedianResp[preSA] )/preMedianResp[preSA] )) \
                / ( (postMedianResp[enInd] - preMedianResp[enInd] ) / preMedianResp[enInd] )
            ax.plot(preSA, y_vals, 'bo') # , markerfacecolor='b'
            # mark the trained odors in red
            ax.plot(enInd, pn, 'ro') # , markerfacecolor='r'
            ax.set_title(r'relative $\Delta$ median')
            ax.set_xlim([0, max(preSA)+1])
            # ax.set_ylim([0,2])
            ax.set_xticks(preSA, minor=False)
            ax.set_xticklabels(classLabels)

            #-------------------------------------------------------------------
            ## means
            # raw means, pre and post
            ax = thisFig.add_subplot(2, 3, 4)
            ax.grid()

            ax.errorbar(preSA, preMeanResp[preSA], yerr=preStdResp[preSA],
                fmt='bo')
            ax.errorbar(postOffset, postMeanResp[postSA], yerr=postStdResp[postSA],
                fmt='bo', markerfacecolor='b')
            ax.errorbar(enInd, preMeanResp[enInd], yerr=preStdResp[enInd], fmt='ro')
            ax.errorbar(enInd + 0.25, postMeanResp[enInd], yerr=postStdResp[enInd],
                fmt='ro', markerfacecolor='r')
            ax.set_title(f'EN {enInd}\n median +/- std')
            ax.set_xlim([0, max(preSA)+1])
            ax.set_ylim([0, 1.1*np.concatenate((preMeanResp, postMeanResp)).max() \
                + np.concatenate((preStdResp, postStdResp)).max()])
            ax.set_xticks(preSA, minor=False)
            ax.set_xticklabels(classLabels)

            # plot spont
            ax.errorbar(preSA[0], preHebMean, yerr=preHebStd, fmt='mo')
            ax.errorbar(postOffset[0], postHebMean, yerr=postHebStd, fmt='mo', markerfacecolor='m')

            # percent change in means
            ax = thisFig.add_subplot(2, 3, 5)
            ax.grid()

            ax.plot(preSA, percentChangeInMeanResp, 'bo', markerfacecolor='b')
            # mark the trained odors in red
            ax.plot(enInd, percentChangeInMeanResp[enInd], 'ro', markerfacecolor='r')
            ax.set_title(r'% $\Delta$ mean')
            ax.set_xlim([0, max(preSA)+1])
            # ax.set_ylim([-50, 1000])
            ax.set_xticks(preSA)
            ax.set_xticklabels(classLabels)

            # relative percent changes
            ax = thisFig.add_subplot(2, 3, 6)
            ax.grid()

            pn = np.sign(postMeanResp[enInd] - preMeanResp[enInd])
            ax.plot(preSA, (pn*percentChangeInMeanResp)/percentChangeInMeanResp[enInd],
                'bo', markerfacecolor='b')
            # mark the trained odors in red
            ax.plot(enInd, pn*1, 'ro', markerfacecolor='r')
            ax.set_title(r'relative $\Delta$ mean')
            ax.set_xlim([0, max(preSA)+1])
            # ax.set_ylim([0, 2])
            ax.set_xticks(preSA)
            ax.set_xticklabels(classLabels)

        # Save plot
        if os.path.isdir(images_folder) and show_acc_plots:
            thisFig.savefig(images_filename + '_en{}.png'.format(enInd), dpi=100)

        #-----------------------------------------------------------------------

        # store results in a list of dicts
        results[enInd]['preTrainOdorResp'] = preTrainOdorResp # preserves all the sniffs for each stimulus
        results[enInd]['postTrainOdorResp'] = postTrainOdorResp
        results[enInd]['preRespSniffsAved'] = preSA # the averaged sniffs for each stimulus
        results[enInd]['postRespSniffsAved'] = postSA
        results[enInd]['odorClass'] = whichClass
        results[enInd]['percentChangeInMeanResp'] = percentChangeInMeanResp # key stat
        results[enInd]['percentChangeInNoiseSubtractedMeanResp'] = percentChangeInNoiseSubtractedMeanResp
        results[enInd]['relativeChangeInNoiseSubtractedMeanResp'] = \
                percentChangeInNoiseSubtractedMeanResp / percentChangeInNoiseSubtractedMeanResp[enInd]
        results[enInd]['percentChangeInMedianResp'] = percentChangeInMedianResp
        results[enInd]['percentChangeInNoiseSubtractedMedianResp'] = percentChangeInNoiseSubtractedMedianResp
        results[enInd]['relativeChangeInNoiseSubtractedMedianResp'] = \
                ( (postMedianResp - preMedianResp - postHebMean )/preMedianResp ) / \
                ( (postMedianResp[enInd] - preMedianResp[enInd] - postHebMean )/preMedianResp[enInd] )
        results[enInd]['trained'] = enInd
        # EN odor responses, pre and post training.
        # these should be vectors of length numStims
        results[enInd]['preMeanResp'] = preMeanResp
        results[enInd]['preStdResp'] = preStdResp
        results[enInd]['postMeanResp'] = postMeanResp
        results[enInd]['postStdResp'] = postStdResp
        # spont responses, pre and post training
        results[enInd]['preSpontMean'] = preSpont.mean()
        results[enInd]['preSpontStd'] = preSpont.std()
        results[enInd]['postSpontMean'] = postSpont.mean()
        results[enInd]['postSpontStd'] = postSpont.std()

    ## Plot EN timecourses normalized by mean digit response

    # labels = whichClass
    if show_time_plots:

        # go through each EN
        for enInd in range(modelParams.nE): # recal EN1 targets digit class 1, EN2 targets digit class 2, etc

            if enInd%3 == 0:
                # make a new figure at ENs 4, 7, 10
                #? test this (below)
                fig_sz = [np.floor(i/100) for i in screen_size]
                enFig = plt.figure(figsize=fig_sz, dpi=100)

            ax = enFig.add_subplot(3, 1, (enInd%3)+1)
            ax.set_xlim([-30, max(simRes['T'])])

            # plot octo
            ax.plot(octoTimes, np.zeros(octoTimes.shape), 'yx')

            # select indices for control
            controlInd = list(range(0, enInd)) + list(range(enInd+1, len(classList)))

            # # plot mean pre and post training of trained digit
            preMean = results[enInd]['preMeanResp']
            # preMeanTr = preMean[enInd]
            preMeanControl = preMean[controlInd].mean()
            # preStd = results[enInd]['preStdResp']
            # preStd = preStd[enInd]
            postMean = results[enInd]['postMeanResp']
            # postMeanTr = postMean[enInd]
            postMeanControl = postMean[controlInd].mean()
            # postStd = results[enInd]['postStdResp']
            # postStd = postStd[enInd]
            preT = simRes['T'] < expP.startTrain
            preTime = simRes['T'][preT]
            preTimeInds = np.nonzero(preT)[0]
            postT = simRes['T'] > expP.endTrain
            postTime = simRes['T'][postT]
            postTimeInds = np.nonzero(postT)[0]
            midT = np.logical_and(simRes['T'] > expP.startTrain, simRes['T'] < expP.endTrain)
            midTime = simRes['T'][midT]
            midTimeInds = np.nonzero(midT)[0]

            # plot ENs

            # normalized by the home class preMean
            ax.plot(preTime, simRes['E'][preTimeInds,enInd] / preMeanControl, color='b')
            # normalized by the home class postMean
            ax.plot(postTime, simRes['E'][postTimeInds,enInd] / postMeanControl, color='b')
            ax.plot(midTime, simRes['E'][midTimeInds,enInd] / 1, color='b')

            # ax.plot(preTime, preMean*np.ones(preTime.shape), color=colors[enInd], '-')
            # ax.plot(postTime, postMean*np.ones(postTime.shape), color=colors[enInd], '-')
            # ax.plot(preTime, (preMean-preStd)*np.ones(preTime.shape), color=colors[enInd], ':')
            # ax.plot(postTime, (postMean-postStd)*np.ones(postTime.shape), color=colors[enInd], ':')

            # plot stims by color
            for i,cl in enumerate(classList):
                classStarts = stimStarts[whichClass == cl]
                ax.plot(classStarts, np.zeros(classStarts.shape), '.', \
                    color=colors[i], markersize=24) # , markerfacecolor=colors[i]

                # reinforce trained color
                if i == enInd:
                    ax.plot(classStarts, 0.001*np.ones(classStarts.shape), '.', \
                        color=colors[i], markersize=24) # , markerfacecolor=colors[i]

            # format
            ax.set_ylim( [0, 1.2* max(simRes['E'][postTimeInds,enInd])/postMeanControl] )
            # rarrow = texlabel('/rarrow')
            ax.set_title(f'EN {enInd} for class {enInd}')

            # Save EN timecourse:
            if os.path.isdir(images_folder) and \
            (enInd%3 == 2 or enInd == (modelParams.nE-1)):
                fig_name = os.path.join(os.getcwd(), images_filename+'_enTimecourses{}.png'.format(enInd))
                enFig.savefig(fig_name, dpi=100)
                print(f'Figure saved: {fig_name}')

    return results

def show_roc_curves(fpr, tpr, roc_auc, class_labels, title_str='', images_filename=''):
    '''
    Plot all ROC curves
    Input: fpr, tpr, roc_auc, images_filename=''
    '''
    from itertools import cycle

    if images_filename:
        images_folder = os.path.dirname(images_filename)

    # create directory for images (if doesnt exist)
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)
        print('Creating results directory: {}'.format(images_filename))

    lw = 1.5

    fig, ax = plt.subplots()
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    for i, color in zip(range(len(class_labels)), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Digit: {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title_str + ': ROC extended to multi-class')
    ax.legend(loc="lower right")

    # Save plot
    if os.path.isdir(images_folder):
        roc_fname = os.path.join(os.getcwd(), images_filename+'.png')
        fig.savefig(roc_fname, dpi=150)
        print(f'ROC curves saved: {roc_fname}')


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
