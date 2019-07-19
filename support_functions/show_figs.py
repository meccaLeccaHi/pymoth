#!/usr/bin/env python3

import numpy as np
import os
import matplotlib.pyplot as plt

def show_FA_thumbs( feature_array, show_per_class, normalize, title_string,
    screen_size, images_filename ):
    '''
    Show thumbnails of inputs used in the experiment.
    Parameters:
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

    _ , _ , num_classes  = feature_array.shape

    total = num_classes*show_per_class # total number of subplots
    num_rows = np.ceil(np.sqrt(total/2)) # n of rows
    num_cols = np.ceil(np.sqrt(total*2)) # n of cols
    vert = 1/(num_rows + 1) # vertical step size
    horiz = 1/(num_cols + 1) # horizontal step size

    fig_sz = [np.floor((i/100)*0.5) for i in screen_size]
    thumbs_fig = plt.figure(figsize=fig_sz, dpi=100)

    for cl in range(num_classes): # 'class' is a keyword in Python; renamed to 'cl'
        for i in range(show_per_class):
            ax_i = show_per_class*(cl) + i + 1
            this_input = feature_array[:, i, cl]

            # renormalize, to offset effect of classMagMatrix scaling
            if normalize:
                this_input /= this_input.max()

            ax_count = i + (cl*num_classes)
            plt.subplot(np.int(num_rows),np.int(num_cols),ax_i)

            side = np.int(np.sqrt(len(this_input)))
            plt.imshow(this_input.reshape((side,side)), cmap='gray', vmin=0, vmax=1)

    # add a title at the bottom
    plt.xlabel(title_string, fontweight='bold')

    # Save plot
    if os.path.isdir(images_folder):
        thumb_name = os.path.join(os.getcwd(), images_filename+'.png')
        thumbs_fig.savefig(thumb_name, dpi=100)
        print(f'Image thumbnails saved: {thumb_name}')
    else:
        print('Image thumbnails NOT SAVED!\nMake sure a valid directory path has been prepended to `images_filename`')

def show_EN_resp( sim_res, model_params, exp_params, show_acc_plots, show_time_plots,
                class_labels, screen_size, images_filename='' ):
    '''
    View readout neurons (EN):
        Color-code them dots by class and by concurrent octopamine.
        Collect stats: median, mean, and std of FR for each digit, pre- and post-training.
        Throughout, digits may be referred to as odors, or as odor puffs.
        'Pre' = naive. 'Post' = post-training

    Parameters:
        1. sim_res: dictionary containing simulation results (output from sdeWrapper)
        2. model_params: object containing model parameters for this moth
        3. exp_params: object containing experiment parameters with timing
            and digit class info from the experiment.
        4. show_acc_plots: show changes in accuracy.
        5. show_time_plots: show EN timecourses.
        6. class_labels: 1 to 10
        7. screen_size: tuple
        8. images_filename: to generate image filenames when saving (includes path).
            Optional argin. If this = '', images will not be saved (ie it's also a flag).

    Returns results list of dictionaries:
        1. pre_mean_resp = numENs x numOdors matrix = mean of EN pre-training
        2. pre_std_resp = numENs x numOdors matrix = std of EN responses pre-training
        3. ditto for post etc
        4. percent_change_mean_resp = 1 x numOdors vector
        5. trained = list of indices corresponding to the odor(s) that were trained
        6. pre_spont_mean = mean(pre_spont)
        7. pre_spont_std = std(pre_spont)
        8. post_spont_mean = mean(post_spont)
        9. post_spont_std = std(post_spont)

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    if images_filename:
        images_folder = os.path.dirname(images_filename)
        # create directory for images (if doesnt exist)
        if not os.path.isdir(images_folder):
            os.mkdir(images_folder)
            print('Creating results directory: {}'.format(images_filename))

    colors = [ (0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 0.3, 0.8), (0.8, 0.3, 1), (0.8, 1, 0.3), (0.5, 0.5, 0.5) ] # for 10 classes
    # concurrent octopamine marked with yellow x's

    if sim_res['octo_hits'].max() > 0:
        octo_times = sim_res['T'][ sim_res['octo_hits'] > 0 ]
    else:
        octo_times = []

    # calc spont stats
    pre_spont = sim_res['E'][ np.logical_and(exp_params.preHebSpontStart < sim_res['T'],
                                    sim_res['T'] < exp_params.preHebSpontStop) ]
    post_spont = sim_res['E'][ np.logical_and(exp_params.postHebSpontStart < sim_res['T'],
                                    sim_res['T'] < exp_params.postHebSpontStop) ]

    pre_heb_mean = pre_spont.mean()
    pre_heb_std = pre_spont.std()
    post_heb_mean = post_spont.mean()
    post_heb_std = post_spont.std()

    ## Set regions to examine:
    # 1. data from exp_params
    # stim_starts = exp_params.stim_starts # to get timeSteps from very start of sim
    stim_starts = exp_params.stimStarts*(exp_params.classMags > 0) # ie only use non-zero puffs
    which_class = exp_params.whichClass*(exp_params.classMags > 0)
    class_list = np.unique(which_class)

    # pre-allocate list of empty dicts
    results = [dict() for i in range(model_params.nE)]

    # Make one stats plot per EN. Loop through ENs:
    for en_ind in range(model_params.nE):

        en_resp = sim_res['E'][:, en_ind]

        ## Calculate pre- and post-train odor response stats
        # Assumes that there is at least 1 sec on either side of an odor without octo

        # pre-allocate for loop
        pre_train_resp = np.full(len(stim_starts), np.nan)
        post_train_resp = np.full(len(stim_starts), np.nan)

        for i, t in enumerate(stim_starts):
            # Note: to find no-octo stim_starts, there is a certain amount of machinery
            # in order to mesh with the timing data from the experiment.
            # For some reason octo_times are not recorded exactly as listed in format
            # short mode. So we need to use abs difference > small thresh, rather
            # than ~ismember(t, octo_times):
            small = 1e-8 # .00000001
            # assign no-octo, PRE-train response val (or -1)
            pre_train_resp[i] = -1 # as flag
            if (len(octo_times)==0) or ((abs(octo_times - t).min() > small) and (t < exp_params.startTrain)):
                resp_ind = np.logical_and(t-1 < sim_res['T'], sim_res['T'] < t+1)
                pre_train_resp[i] = en_resp[resp_ind].max()

            # assign no-octo, POST-train response val (or -1)
            post_train_resp[i] = -1
            if len(octo_times)!=0:
                if (abs(octo_times - t).min() > small) and (t > exp_params.endTrain):
                    resp_ind = np.logical_and(t-1 < sim_res['T'], sim_res['T'] < t+1)
                    post_train_resp[i] = en_resp[resp_ind].max()

        # pre-allocate for loop
        pre_mean_resp, pre_median_resp, pre_std_resp, pre_num_puffs, post_mean_resp, \
            post_median_resp, post_std_resp, post_num_puffs = \
            [np.full(len(class_list), np.nan) for _ in range(8)]

        # calc no-octo stats for each odor, pre and post train:
        for k, cl in enumerate(class_list):
            current_class = which_class==cl
            pre_SA = pre_train_resp[np.logical_and(pre_train_resp>=0, current_class)]
            post_SA = post_train_resp[np.logical_and(post_train_resp>=0, current_class)]

            ## calculate the averaged sniffs of each sample: SA means 'sniffsAveraged'
            # this will contain the average responses over all sniffs for each sample
            # DEV NOTE: Changed pretty drastically from orig version, but should be the same.
            # Double check with CBD
            if len(pre_SA)==0:
                pre_mean_resp[k] = -1
                pre_median_resp[k] = -1
                pre_std_resp[k] = -1
                pre_num_puffs[k] = 0
            else:
                pre_mean_resp[k] = pre_SA.mean()
                pre_median_resp[k] = np.median(pre_SA)
                pre_std_resp[k] = pre_SA.std()
                pre_num_puffs[k] = len(pre_SA)

            if len(post_SA)==0:
                post_mean_resp[k] = -1
                post_median_resp[k] = -1
                post_std_resp[k] = -1
                post_num_puffs[k] = 0
            else:
                post_mean_resp[k] = post_SA.mean()
                post_median_resp[k] = np.median(post_SA)
                post_std_resp[k] = post_SA.std()
                post_num_puffs[k] = len(post_SA)

        # # to plot +/- 1 std of % change in mean_resp, we want the std of our
        # # estimate of the mean = std_resp/sqrt(numPuffs). Make this calc:
        # pre_std_mean_est = pre_std_resp/np.sqrt(pre_num_puffs)
        # post_std_mean_est = post_std_resp/np.sqrt(post_num_puffs)

        pre_SA = np.nonzero(pre_num_puffs > 0)[0]
        post_SA = np.nonzero(post_num_puffs > 0)[0]
        post_offset = post_SA + 0.25

        percent_change_mean_resp = (100*(post_mean_resp[pre_SA] - pre_mean_resp[pre_SA]))\
                                    /pre_mean_resp[pre_SA]
        percent_change_noise_sub_mean_resp = \
                                (100*(post_mean_resp[pre_SA] - pre_mean_resp[pre_SA] - post_heb_mean))\
                                /pre_mean_resp[pre_SA]

        percent_change_median_resp = (100*(post_median_resp[pre_SA] - pre_median_resp[pre_SA]))\
                                /pre_median_resp[pre_SA]
        percent_change_noise_sub_median_resp = \
                                (100*(post_median_resp[pre_SA] - pre_median_resp[pre_SA] - post_heb_mean))\
                                /pre_median_resp[pre_SA]

        # plot stats if wished:
        if show_acc_plots:

            fig = show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp,
                pre_std_resp, post_offset, post_mean_resp, post_median_resp, post_std_resp,
                class_labels, pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std,
                percent_change_mean_resp, screen_size)

        # Save plot
        if os.path.isdir(images_folder) and show_acc_plots:
            fig.savefig(images_filename + '_en{}.png'.format(en_ind), dpi=100)

        #-----------------------------------------------------------------------

        # store results in a list of dicts
        results[en_ind]['pre_train_resp'] = pre_train_resp # preserves all the sniffs for each stimulus
        results[en_ind]['post_train_resp'] = post_train_resp
        results[en_ind]['preRespSniffsAved'] = pre_SA # the averaged sniffs for each stimulus
        results[en_ind]['postRespSniffsAved'] = post_SA
        results[en_ind]['odor_class'] = which_class
        results[en_ind]['percent_change_mean_resp'] = percent_change_mean_resp # key stat
        results[en_ind]['percent_change_noise_sub_mean_resp'] = percent_change_noise_sub_mean_resp
        results[en_ind]['relativeChangeInNoiseSubtractedMeanResp'] = \
                percent_change_noise_sub_mean_resp / percent_change_noise_sub_mean_resp[en_ind]
        results[en_ind]['percent_change_median_resp'] = percent_change_median_resp
        results[en_ind]['percent_change_noise_sub_median_resp'] = percent_change_noise_sub_median_resp
        results[en_ind]['relativeChangeInNoiseSubtractedMedianResp'] = \
                ( (post_median_resp - pre_median_resp - post_heb_mean )/pre_median_resp ) / \
                ( (post_median_resp[en_ind] - pre_median_resp[en_ind] - post_heb_mean )/pre_median_resp[en_ind] )
        results[en_ind]['trained'] = en_ind
        # EN odor responses, pre and post training.
        # these should be vectors of length numStims
        results[en_ind]['pre_mean_resp'] = pre_mean_resp
        results[en_ind]['pre_std_resp'] = pre_std_resp
        results[en_ind]['post_mean_resp'] = post_mean_resp
        results[en_ind]['post_std_resp'] = post_std_resp
        # spont responses, pre and post training
        results[en_ind]['pre_spont_mean'] = pre_spont.mean()
        results[en_ind]['pre_spont_std'] = pre_spont.std()
        results[en_ind]['post_spont_mean'] = post_spont.mean()
        results[en_ind]['post_spont_std'] = post_spont.std()

    ## Plot EN timecourses normalized by mean digit response
    if show_time_plots:

        # go through each EN
        for en_ind in range(model_params.nE): # recal EN1 targets digit class 1, EN2 targets digit class 2, etc

            if en_ind%3 == 0:
                # make a new figure at ENs 4, 7, 10
                fig_sz = [np.floor(i/100) for i in screen_size]
                fig = plt.figure(figsize=fig_sz, dpi=100)

            ax = fig.add_subplot(3, 1, (en_ind%3)+1)

            show_en_timecourse(ax, en_ind, sim_res, octo_times, class_list, results,
                exp_params, stim_starts, which_class, colors )

            # Save EN timecourse:
            if os.path.isdir(images_folder) and \
            (en_ind%3 == 2 or en_ind == (model_params.nE-1)):
                fig_name = os.path.join(os.getcwd(), images_filename+'_en_timecourses{}.png'.format(en_ind))
                fig.savefig(fig_name, dpi=100)
                print(f'Figure saved: {fig_name}')

    return results

def plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str,
    y_axis_label=True, legend=True):

    from itertools import cycle

    lw = 1.5

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
    if y_axis_label:
        ax.set_ylabel('True Positive Rate')
    ax.set_title(title_str + ': ROC extended to multi-class')
    if legend:
        ax.legend(loc="lower right")
    # return ax

def show_roc_curves(fpr, tpr, roc_auc, class_labels, title_str='', images_filename=''):
    '''
    Plot all ROC curves
    Parameters: fpr, tpr, roc_auc, images_filename=''
    '''

    if images_filename:
        images_folder = os.path.dirname(images_filename)
        # create directory for images (if doesnt exist)
        if not os.path.isdir(images_folder):
            os.mkdir(images_folder)
            print('Creating results directory: {}'.format(images_filename))

    fig, ax = plt.subplots()
    ax = plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str)

    # Save plot
    if os.path.isdir(images_folder):
        roc_fname = os.path.join(os.getcwd(), images_filename+'.png')
        fig.savefig(roc_fname, dpi=150)
        print(f'ROC curves saved: {roc_fname}')

def show_roc_subplots(roc_dict_list, title_str_list, class_labels, images_filename=''):

    if images_filename:
        images_folder = os.path.dirname(images_filename)
        # create directory for images (if doesnt exist)
        if images_folder and not os.path.isdir(images_folder):
            os.mkdir(images_folder)
            print('Creating results directory: {}'.format(images_filename))

    fig, axes = plt.subplots(1, len(roc_dict_list), figsize=(15,5), sharey=True)

    y_ax_list = [True, False, False]
    legend_list = [True, False, False]

    for i in range(len(roc_dict_list)):
        ax = axes[i]
        fpr = roc_dict_list[i]['fpr']
        tpr = roc_dict_list[i]['tpr']
        roc_auc = roc_dict_list[i]['roc_auc']
        title_str = title_str_list[i]

        plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str,
            y_axis_label=y_ax_list[i], legend=legend_list[i])

    fig.tight_layout()

    # save plot
    if os.path.isdir(images_folder):
        roc_fname = os.path.join(os.getcwd(), images_filename+'.png')
        fig.savefig(roc_fname, dpi=150)
        print(f'ROC curves saved: {roc_fname}')
    else:
        print('ROC curves NOT SAVED!\nMake sure a valid directory path has been prepended to `images_filename`')

def show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp, pre_std_resp,
    post_offset, post_mean_resp, post_median_resp, post_std_resp, class_labels,
    pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std, percent_change_mean_resp, screen_size):
    '''
    show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp, pre_std_resp,
        post_offset, post_mean_resp, post_median_resp, post_std_resp, class_labels,
        pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std, percent_change_mean_resp, screen_size)
    '''
    fig_sz = [np.floor((i/100)*0.8) for i in screen_size]
    fig = plt.figure(figsize=fig_sz, dpi=100)

    # medians, pre and post
    ax = fig.add_subplot(2, 3, 1)
    ax.grid()
    ax.plot(pre_SA, pre_median_resp[pre_SA], '*b')
    ax.plot(post_offset, post_median_resp[post_SA], 'bo') # , markerfacecolor='b'
    #   ax.plot(pre, pre_mean_resp + pre_std_resp, '+g')
    #   ax.plot(post, post_mean_resp + post_std_resp, '+g')
    #   ax.plot(pre, pre_mean_resp - pre_std_resp, '+g')
    #   ax.plot(post, post_mean_resp - post_std_resp, '+g')

    # make the home EN of this plot red
    ax.plot(en_ind, pre_median_resp[en_ind], 'ro')
    ax.plot(en_ind + 0.25, post_median_resp[en_ind], 'ro') # ,'markerfacecolor','r'
    ax.set_title(f'EN {en_ind}\n median +/- std')
    ax.set_xlim([0, max(pre_SA) + 1])
    ax.set_ylim([0, 1.1*max(np.concatenate((pre_median_resp, post_median_resp)))])
    ax.set_xticks(pre_SA, minor=False)
    ax.set_xticklabels(class_labels)

    # connect pre to post with lines for clarity
    for j in range(len(pre_SA)):
        if j==1:
            line_color = 'r'
        else:
            line_color = 'b'
        ax.plot(
                [pre_SA[j], post_offset[j]],
                [pre_median_resp[pre_SA[j]], post_median_resp[pre_SA[j]]],
                line_color
                )

    # percent change in medians
    ax = fig.add_subplot(2, 3, 2)
    ax.grid()
    ax.plot(
        pre_SA,
        (100*(post_median_resp[pre_SA] - pre_median_resp[pre_SA]))/pre_median_resp[pre_SA],
        'bo') # , markerfacecolor='b'

    # mark the trained odors in red:
    ax.plot(
        en_ind,
        (100*(post_median_resp[en_ind] - pre_median_resp[en_ind]))/pre_median_resp[en_ind],
        'ro') # , markerfacecolor='r'
    ax.set_title(r'% $\Delta$ median')
    ax.set_xlim([0, max(pre_SA)+1])
    # ax.set_ylim([-50,400])
    ax.set_xticks(pre_SA, minor=False)
    ax.set_xticklabels(class_labels)

    # relative changes in median, ie control/trained
    ax = fig.add_subplot(2, 3, 3)
    ax.grid()
    pn = np.sign(post_median_resp[en_ind] - pre_median_resp[en_ind])
    y_vals = (pn * ( (post_median_resp[pre_SA] - pre_median_resp[pre_SA] )/pre_median_resp[pre_SA] )) \
        / ( (post_median_resp[en_ind] - pre_median_resp[en_ind] ) / pre_median_resp[en_ind] )
    ax.plot(pre_SA, y_vals, 'bo') # , markerfacecolor='b'
    # mark the trained odors in red
    ax.plot(en_ind, pn, 'ro') # , markerfacecolor='r'
    ax.set_title(r'relative $\Delta$ median')
    ax.set_xlim([0, max(pre_SA)+1])
    # ax.set_ylim([0,2])
    ax.set_xticks(pre_SA, minor=False)
    ax.set_xticklabels(class_labels)

    #-------------------------------------------------------------------
    ## means
    # raw means, pre and post
    ax = fig.add_subplot(2, 3, 4)
    ax.grid()

    ax.errorbar(pre_SA, pre_mean_resp[pre_SA], yerr=pre_std_resp[pre_SA],
        fmt='bo')
    ax.errorbar(post_offset, post_mean_resp[post_SA], yerr=post_std_resp[post_SA],
        fmt='bo', markerfacecolor='b')
    ax.errorbar(en_ind, pre_mean_resp[en_ind], yerr=pre_std_resp[en_ind], fmt='ro')
    ax.errorbar(en_ind + 0.25, post_mean_resp[en_ind], yerr=post_std_resp[en_ind],
        fmt='ro', markerfacecolor='r')
    ax.set_title(f'EN {en_ind}\n median +/- std')
    ax.set_xlim([0, max(pre_SA)+1])
    ax.set_ylim([0, 1.1*np.concatenate((pre_mean_resp, post_mean_resp)).max() \
        + np.concatenate((pre_std_resp, post_std_resp)).max()])
    ax.set_xticks(pre_SA, minor=False)
    ax.set_xticklabels(class_labels)

    # plot spont
    ax.errorbar(pre_SA[0], pre_heb_mean, yerr=pre_heb_std, fmt='mo')
    ax.errorbar(post_offset[0], post_heb_mean, yerr=post_heb_std, fmt='mo', markerfacecolor='m')

    # percent change in means
    ax = fig.add_subplot(2, 3, 5)
    ax.grid()

    ax.plot(pre_SA, percent_change_mean_resp, 'bo', markerfacecolor='b')
    # mark the trained odors in red
    ax.plot(en_ind, percent_change_mean_resp[en_ind], 'ro', markerfacecolor='r')
    ax.set_title(r'% $\Delta$ mean')
    ax.set_xlim([0, max(pre_SA)+1])
    # ax.set_ylim([-50, 1000])
    ax.set_xticks(pre_SA)
    ax.set_xticklabels(class_labels)

    # relative percent changes
    ax = fig.add_subplot(2, 3, 6)
    ax.grid()

    pn = np.sign(post_mean_resp[en_ind] - pre_mean_resp[en_ind])
    ax.plot(pre_SA, (pn*percent_change_mean_resp)/percent_change_mean_resp[en_ind],
        'bo', markerfacecolor='b')
    # mark the trained odors in red
    ax.plot(en_ind, pn*1, 'ro', markerfacecolor='r')
    ax.set_title(r'relative $\Delta$ mean')
    ax.set_xlim([0, max(pre_SA)+1])
    # ax.set_ylim([0, 2])
    ax.set_xticks(pre_SA)
    ax.set_xticklabels(class_labels)
    return fig

def show_en_timecourse(ax, en_ind, sim_res, octo_times, class_list, results,
    exp_params, stim_starts, which_class, colors ):

    ax.set_xlim([-30, max(sim_res['T'])])

    # plot octo
    ax.plot(octo_times, np.zeros(octo_times.shape), 'yx')

    # select indices for control
    control_ind = list(range(0, en_ind)) + list(range(en_ind+1, len(class_list)))

    # # plot mean pre and post training of trained digit
    pre_mean = results[en_ind]['pre_mean_resp']
    # pre_meanTr = pre_mean[en_ind]
    pre_mean_control = pre_mean[control_ind].mean()
    # pre_std = results[en_ind]['pre_std_resp']
    # pre_std = pre_std[en_ind]
    post_mean = results[en_ind]['post_mean_resp']
    post_mean_control = post_mean[control_ind].mean()
    # post_std = results[en_ind]['post_std_resp']
    # post_std = post_std[en_ind]
    pre_t = sim_res['T'] < exp_params.startTrain
    pre_time = sim_res['T'][pre_t]
    pre_time_inds = np.nonzero(pre_t)[0]
    post_t = sim_res['T'] > exp_params.endTrain
    post_time = sim_res['T'][post_t]
    post_time_inds = np.nonzero(post_t)[0]
    mid_t = np.logical_and(sim_res['T'] > exp_params.startTrain, sim_res['T'] < exp_params.endTrain)
    mid_time = sim_res['T'][mid_t]
    mid_time_inds = np.nonzero(mid_t)[0]

    ## plot ENs
    # normalized by the home class pre_mean
    ax.plot(pre_time, sim_res['E'][pre_time_inds,en_ind] / pre_mean_control, color='b')
    # normalized by the home class post_mean
    ax.plot(post_time, sim_res['E'][post_time_inds,en_ind] / post_mean_control, color='b')
    ax.plot(mid_time, sim_res['E'][mid_time_inds,en_ind] / 1, color='b')

    # ax.plot(pre_time, pre_mean*np.ones(pre_time.shape), color=colors[en_ind], '-')
    # ax.plot(post_time, post_mean*np.ones(post_time.shape), color=colors[en_ind], '-')
    # ax.plot(pre_time, (pre_mean-pre_std)*np.ones(pre_time.shape), color=colors[en_ind], ':')
    # ax.plot(post_time, (post_mean-post_std)*np.ones(post_time.shape), color=colors[en_ind], ':')

    # plot stims by color
    for i,cl in enumerate(class_list):
        class_starts = stim_starts[which_class == cl]
        ax.plot(class_starts, np.zeros(class_starts.shape), '.', \
            color=colors[i], markersize=24) # , markerfacecolor=colors[i]

        # reinforce trained color
        if i == en_ind:
            ax.plot(class_starts, 0.001*np.ones(class_starts.shape), '.', \
                color=colors[i], markersize=24) # , markerfacecolor=colors[i]

    # format
    ax.set_ylim( [0, 1.2* max(sim_res['E'][post_time_inds,en_ind])/post_mean_control] )
    # rarrow = texlabel('/rarrow')
    ax.set_title(f'EN {en_ind} for class {en_ind}')

    return ax

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
