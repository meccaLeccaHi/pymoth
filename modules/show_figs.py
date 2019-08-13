#!/usr/bin/env python3

"""

.. module:: show_figs
   :platform: Unix
   :synopsis: Figure generation module.

.. moduleauthor:: Adam P. Jones <ajones173@gmail.com>

"""

import numpy as np
import os
import matplotlib.pyplot as plt

def show_FA_thumbs( feature_array, show_per_class, normalize, title_string,
    screen_size, images_filename ):
    """

    Show thumbnails of inputs used in the experiment.

    Args:
        feature_array (numpy array): either 3-D (1 = cols of features, 2 = within class samples, 3 = class) \
                or 2-D (1 = cols of features, 2 = within class samples, no 3)
        show_per_class (int): how many of the thumbnails from each class to show.
        normalize (bool): 1 to rescale thumbs to [0 1], 0 to not
        title_string (str): string for figure title
        screen_size (tuple): width, height
        images_filename (str): including absolute path

    Returns
    -------
        None

    >>> show_FA_thumbs(_thumb_array, 1, 1, 'Input thumbnails', (1920,1080), 'foo/thumbnails')

    """

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
        thumb_name = images_filename + '.png'
        thumbs_fig.savefig(thumb_name, dpi=100)
        print(f'Image thumbnails saved: {thumb_name}')
    else:
        print('Image thumbnails NOT SAVED!',
            'Make sure a valid (relative) directory path has been prepended to `images_filename`')


def plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str,
    y_axis_label=True, legend=True):
    """
    plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str, \
    y_axis_label=True, legend=True)

    Args:
        ax (object): matplotlib axis (ie subplot)
        tpr (dict): true-positive rate for each class
        fpr (dict): false-positive rate for each class
        roc_auc (dict): ROC AUC for each class
        class_labels (numpy array): class labels (0:9 for MNIST)
        title_str (str): string to use in the title for this particular subplot
        y_axis_label(str): string to use in the y-axis label for this particular \
        subplot
        legend (bool): toggle legend for figure

    Returns
    -------
        None

    >>> plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str, \
    y_axis_label='foo', legend=True)
    """
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

def show_roc_curves(tpr, fpr, roc_auc, class_labels, title_str='', images_filename=''):
    """
    Compute macro-average ROC curves and plot.

    Args:
        tpr (dict): true-positive rate for each class
        fpr (dict): false-positive rate for each class
        roc_auc (dict): ROC AUC for each class
        class_labels (numpy array): class labels (0:9 for MNIST)
        title_str (str): string to use in the title for this particular subplot
        images_filename (str): directory to save figure output

    Returns
    -------
        None

    >>> show_roc_curves(roc_knn['tpr'], roc_knn['fpr'], roc_knn['roc_auc'], \
    class_labels, title_str='KNN', images_filename='dirname/filename')
    """

    if images_filename:
        images_folder = os.path.dirname(images_filename)
        # create directory for images (if doesnt exist)
        if not os.path.isdir(images_folder):
            os.mkdir(images_folder)
            print('Creating results directory: {}'.format(images_folder))

    fig, ax = plt.subplots()
    plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str)

    # Save plot
    if os.path.isdir(images_folder):
        roc_filename = images_filename + '_ROC_' + title_str + '.png'
        fig.savefig(roc_filename, dpi=150)
        print(f'Figure saved: {roc_filename}')

def show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp, pre_std_resp,
    post_offset, post_mean_resp, post_median_resp, post_std_resp, class_labels,
    pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std, percent_change_mean_resp, screen_size):
    """
    Plot accuracy of MothNet for each class.

    Args:
        pre_SA (numpy array): pre-sniff average,
        post_SA (numpy array): post-sniff average,
        en_ind (int): index of current EN,
        pre_mean_resp (numpy array): mean resp before training [#classes x 1],
        pre_median_resp (numpy array): median resp before training [#classes x 1],
        pre_std_resp (numpy array): std of resp before training [#classes x 1],
        post_offset (numpy array): pre-sniff average plus offset value,
        post_mean_resp (numpy array): mean resp after training [#classes x 1],
        post_median_resp (numpy array): median resp after training [#classes x 1],
        post_std_resp (numpy array): std of resp after training [#classes x 1],
        class_labels (numpy array): class labels (0:9 for MNIST),
        pre_heb_mean: mean spontaneous activity before training [#classes x 1],
        pre_heb_std: std of spontaneous activity before training [#classes x 1],
        post_heb_mean: mean spontaneous activity after training [#classes x 1],
        post_heb_std: std of spontaneous activity after training [#classes x 1],
        percent_change_mean_resp: mean response converted to percent change [#classes x 1],
        screen_size (tuple): [optional] screen size (width, height) for images,

    Returns
    -------
        fig (object)
            matplotlib figure handle

    >>> show_acc(pre_SA, post_SA, en_ind, pre_mean_resp, pre_median_resp, pre_std_resp, \
        post_offset, post_mean_resp, post_median_resp, post_std_resp, class_labels, \
        pre_heb_mean, pre_heb_std, post_heb_mean, post_heb_std, percent_change_mean_resp, \
        screen_size)
    """
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

def show_timecourse(ax, en_ind, sim_results, octo_times, class_list, results,
    exp_params, stim_starts, which_class ):
    """
    Plot the timecourse of EN responses of MothNet.

    Args:
        ax (object): matplotlib axis,
        en_ind (int): index of current EN,
        sim_results (dict): simulation results (output from :func:`sde_wrap`),
        octo_times (numpy array): timing of octopamine,
        class_labels (numpy array): labels, eg 0:9 for MNIST,
        results (list): list of dicts containing simulation results (output from \
        :func:`sde_wrap`),
        exp_params (class): timing info about experiment, eg when stimuli are given,
        stim_starts (numpy array): time-steps for current stimuli,
        which_class (numpy array): classes for current stimuli,

    Returns
    -------
        ax (object)
            matplotlib axis

    >>> show_timecourse(ax, en_ind, sim_results, octo_times, class_list, results, \
        exp_params, stim_starts, which_class )
    """
    colors = [ (0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1),
         (1, 0.3, 0.8), (0.8, 0.3, 1), (0.8, 1, 0.3), (0.5, 0.5, 0.5) ] # for 10 classes

    ax.set_xlim([-30, max(sim_results['T'])])

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
    pre_t = sim_results['T'] < exp_params.startTrain
    pre_time = sim_results['T'][pre_t]
    pre_time_inds = np.nonzero(pre_t)[0]
    post_t = sim_results['T'] > exp_params.endTrain
    post_time = sim_results['T'][post_t]
    post_time_inds = np.nonzero(post_t)[0]
    mid_t = np.logical_and(sim_results['T'] > exp_params.startTrain,
        sim_results['T'] < exp_params.endTrain)
    mid_time = sim_results['T'][mid_t]
    mid_time_inds = np.nonzero(mid_t)[0]

    ## plot ENs
    # normalized by the home class pre_mean
    ax.plot(pre_time, sim_results['E'][pre_time_inds,en_ind] / pre_mean_control, color='b')
    # normalized by the home class post_mean
    ax.plot(post_time, sim_results['E'][post_time_inds,en_ind] / post_mean_control, color='b')
    ax.plot(mid_time, sim_results['E'][mid_time_inds,en_ind] / 1, color='b')

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
    ax.set_ylim( [0, 1.2* max(sim_results['E'][post_time_inds,en_ind])/post_mean_control] )
    # rarrow = texlabel('/rarrow')
    ax.set_title(f'EN {en_ind} for class {en_ind}')

    return ax

def show_multi_roc(self, model_names, class_labels, images_filename=''):
    """
    Show ROC plot for each model in a subplot of a single figure.

    Args:
        model_names (list): names (strings) of models being plotted,
        class_labels (numpy array): label for each class of current stimuli,
        images_filename (str): [optional] name to use for image filename,

    Returns
    -------
        None

    >>> show_multi_roc(model_names, class_labels, images_filename='foo')
    """

    if images_filename:
        images_folder = os.path.dirname(images_filename)

        # create directory for images (if doesnt exist)
        if images_folder and not os.path.isdir(images_folder):
            os.mkdir(images_folder)
            print('Creating results directory: {}'.format(images_folder))

    roc_dict_list = [self.output_trained_log_loss, self.roc_svm, self.roc_knn]

    fig, axes = plt.subplots(1, len(roc_dict_list), figsize=(15,5), sharey=True)

    y_ax_list = [True, False, False]
    legend_list = [True, False, False]

    for i in range(len(roc_dict_list)):
        ax = axes[i]
        fpr = roc_dict_list[i]['fpr']
        tpr = roc_dict_list[i]['tpr']
        roc_auc = roc_dict_list[i]['roc_auc']
        title_str = model_names[i]

        plot_roc_multi(ax, fpr, tpr, roc_auc, class_labels, title_str,
            y_axis_label=y_ax_list[i], legend=legend_list[i])

    fig.tight_layout()

    # save plot
    if os.path.isdir(images_folder):
        roc_filename = images_filename + '.png'
        fig.savefig(roc_filename, dpi=150)
        print(f'Figure saved: {roc_filename}')
    else:
        print('ROC curves NOT SAVED!\nMake sure a valid directory path has been prepended to `images_filename`')

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
