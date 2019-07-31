#!/usr/bin/env python3

import numpy as np
import itertools

def confusion_matrix( true_classes, pred_classes ):
    '''
    Calculate confusion matrix
    ex: confusion = confusion_matrix(true_classes, pred_classes)

    Params:
    - True classes (class labels i.e. 'y')
    - Predicted classes (predicted labels i.e. 'y-hat')
    Returns:
    - Confusion matrix as Numpy array
    '''

    class_members = list(set(true_classes))

    # create matrix of (true,pred) indices
    ind_mat = list(itertools.product(class_members, class_members))
    conf_mat = []
    for i in ind_mat:
        true_N = [sample==i[0] for sample in true_classes]
        pred_N = [sample==i[1] for sample in pred_classes]
        conf_mat.append(sum([j&k for j,k in zip(true_N,pred_N)]))

    # reshape to square based on number of different classes
    return np.reshape(np.array(conf_mat), (-1, len(set(true_classes))))

def roc_multi(true_classes, likelihoods):
    '''
    Measure ROC AUC for multi-class classifiers.
    Params:
    - true_classes [np array - shape: (observations,)]
    - likelihoods [np array - shape: (observations, classes)]
    '''

    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    n_classes = len(set(true_classes))

    # one-hot-encode target labels
    targets = true_classes.astype(int)
    targets = np.eye(n_classes)[targets]

    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in set(true_classes.astype(int)):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], likelihoods[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(), likelihoods.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ## compute macro-average ROC curve and ROC area
    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # finally, average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    output = dict()
    output['targets'] = targets
    output['roc_auc'] = roc_auc
    output['fpr'] = fpr
    output['tpr'] = tpr
    return output

def classify_digits_log_likelihood( results ):
    '''
    function [output] = classify_digits_log_likelihood(results)
    Classify the test digits in a run using log likelihoods from the various EN responses:
    Parameters:
    results = 1 x 10 struct produced by viewENresponses.
        i'th entry gives results for all classes, in the i'th EN
    Important fields:
    1. post_mean_resp, postStdResp (to calculate post-training, ie val, digit response distributions)
    2. postTrainOdorResponse (gives the actual responses for each val digit, for that EN)
        Note that non-post-train odors have response = -1 as a flag
    3. odorClass: gives the true labels of each digit, 0 to 9 (10 = '0'). this is the same in each EN

    output:
    - output = struct with the following fields:
        1. likelihoods = n x 10 matrix, each row a postTraining digit (entries are summed log likelihoods)
        2. true_classes = shortened version of whichOdor (with only postTrain, ie validation, entries)
        3. pred_classes = predicted classes
        4. conf_mat = raw counts, rows = ground truth, cols = predicted
        5. class_acc = 1 x 10 vector, with class accuracies as percentages
        6. total_acc = overall accuracy as percentage

    ------------------------------------------------------------------------------

    plan:
    1. for each test digit (ignore non-postTrain digits), for each EN, calculate the # stds from the test
    digit is from each class distribution. This makes a 10 x 10 matrix
    where each row corresponds to an EN, and each column corresponds to a class.
    2. Square this matrix by entry. Sum the columns. Select the col with the lowest value as the predicted
    class. Return the vector of sums in 'likelihoods'.
    3. The rest is simple calculation.

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    nEn = len(results) # number of ENs, same as number of classes
    pre_train_inds = np.nonzero(results[1]['post_train_resp'] >= 0)[0] # indices of post-train (ie validation) digits
    # DEV NOTE: Why use 2 (1, here) as index above? Ask CBD
    n_post = len(pre_train_inds) # number of post-train digits

    # extract true classes:
    true_classes = results[0]['odor_class'][pre_train_inds] # throughout, digits may be referred to as odors or 'odor puffs'
    # DEV NOTE: Why use 1 (0, here) as index above? Ask CBD

    # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    post_train_resp = np.full((nEn,n_post), np.nan)
    for i,resp in enumerate(results):
        post_train_resp[i,:] = resp['post_train_resp'][pre_train_inds]

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class:
    mu = np.full((nEn,nEn), np.nan)
    sig = np.full((nEn,nEn), np.nan)
    for i,resp in enumerate(results):
        mu[i,:] = resp['post_mean_resp']
        sig[i,:] = resp['post_std_resp']

    # for each EN:
    # get the likelihood of each puff (ie each col of post_train_resp)
    likelihoods = np.zeros((n_post,nEn))
    for i in range(n_post):
        # Caution: post_train_resp[:,i] becomes a row vector, but we need it to stay as a
        # col vector so we can make 10 identical columns. So transpose it back with [np.newaxis]
        a = post_train_resp[:,i][np.newaxis]
        dist = ( np.tile( a.T, ( 1, 10 )) - mu) / sig # 10 x 10 matrix
        # The ith row, jth col entry is the mahalanobis distance of this test
        # digit's response from the i'th ENs response to the j'th class.
        # For example, the diagonal contains the mahalanobis distance of this
        # digit's response to each EN's home-class response.

        likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener

    # make predictions:
    pred_classes = np.argmin(likelihoods, axis=1)

    # calc accuracy percentages:
    class_acc = np.zeros(nEn)
    for i in range(nEn):
        class_acc[i] = (100*np.logical_and(pred_classes == i, true_classes == i).sum())/(true_classes == i).sum()

    total_acc = (100*(pred_classes == true_classes).sum())/len(true_classes)

    # calc confusion matrix:
    # i,j'th entry is number of test digits with true label i that were predicted to be j
    confusion = confusion_matrix(true_classes, pred_classes)

    # measure ROC AUC for each class
    roc_dict = roc_multi(true_classes, likelihoods*-1)

    output = dict()
    output['true_classes'] = true_classes
    output['targets'] = roc_dict['targets']
    output['roc_auc'] = roc_dict['roc_auc']
    output['fpr'] = roc_dict['fpr']
    output['tpr'] = roc_dict['tpr']
    output['true_classes']
    output['pred_classes'] = pred_classes
    output['likelihoods'] = likelihoods
    output['acc_perc'] = class_acc
    output['total_acc'] = total_acc
    output['conf_mat'] = confusion

    return output


def classify_digits_thresholding(results, home_advantage, home_thresh_sigmas, above_home_thresh_reward):
    '''
    Classify the test digits in a run using log likelihoods from the various EN
    responses, with the added option of rewarding high scores relative to an ENs
    home-class expected response distribution.
    One use of this function is to apply de-facto thresholding on discrete ENs,
    so that the predicted class corresponds to the EN that spiked most strongly
    (relative to its usual home-class response).
    Parameters:
     1. results = 1 x 10 struct produced by viewENresponses. i'th entry gives
     results for all classes, in the i'th EN.
      Important fields:
        a. post_mean_resp, postStdResp (to calculate post-training, ie val, digit
        response distributions).
        b. postTrainOdorResponse (gives the actual responses for each val digit,
        for that EN)
            Note that non-post-train odors have response = -1 as a flag.
        c. odorClass: gives the true labels of each digit, 1 to 10 (10 = '0').
        This is the same in each EN.
     2. 'home_advantage' is the emphasis given to the home EN. It
            multiplies the off-diagonal of dist. 1 -> no advantage (default).
            Very high means that a test digit will be classified according to
            the home EN it does best in, ie each EN acts on it's own.
     3. 'home_thresh_sigmas' = the number of stds below an EN's home-class mean
            that we set a threshold, such that if a digit scores above this threshold
            in an EN, that EN will be rewarded by 'above_home_thresh_reward' (below)
     4. 'above_home_thresh_reward': if a digit's response scores above the EN's mean
            home-class value, reward it by dividing by above_home_thresh_reward. This
            reduces the log likelihood score for that EN.

    Returns:
        - A dictionary with the following fields:
            1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries
            are summed log likelihoods.
            2. true_classes = shortened version of whichOdor (with only postTrain, ie
            validation, entries)
            3. pred_classes = predicted classes
            4. conf_mat = raw counts, rows = ground truth, cols = predicted
            5. class_acc = 1 x 10 vector, with class accuracies as percentages
            6. total_acc = overall accuracy as percentage
    ----------------------------------------------------------------------------

    plan:
    1. for each test digit (ignore non-postTrain digits), for each EN, calculate
        the # stds from the test digit is from each class distribution. This makes
        a 10 x 10 matrix where each row corresponds to an EN, and each column
        corresponds to a class.
    2. Square this matrix by entry. Sum the columns. Select the col with the
        lowest value as the predicted class. Return the vector of sums in 'likelihoods'.
    3. The rest is simple calculation.

    The following values of argin2,3,4 correspond to the log likelihood
    classifier in 'classifyDigitsViaLogLikelihood.m':
        home_advantage = 1
        home_thresh_sigmas = any number
        above_home_thresh_reward = 1
    The following value enables pure home-class thresholding:
        home_advantage = 1e12 # to effectively eliminate off-diagonals

    Copyright (c) 2019 Adam P. Jones (ajones173@gmail.com) and Charles B. Delahunt (delahunt@uw.edu)
    MIT License
    '''

    nEn = len(results) # number of ENs, same as number of classes
    pre_train_inds = np.nonzero(results[1]['post_train_resp'] >= 0)[0] # indices of post-train (ie validation) digits
    # DEV NOTE: Why use 2 (1, in Python) as index above? Ask CBD
    n_post = len(pre_train_inds) # number of post-train digits

    # extract true classes:
    true_classes = results[0]['odor_class'][pre_train_inds] # throughout, digits may be referred to as odors or 'odor puffs'
    # DEV NOTE: Why use 1 (0, in Python) as index above? Ask CBD

    # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    post_train_resp = np.full((nEn,n_post), np.nan)
    for i,resp in enumerate(results):
        post_train_resp[i,:] = resp['post_train_resp'][pre_train_inds]

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class.
    # For example, the i'th row, j'th col entry of 'mu' is the mean of the i'th
    # EN in response to digits from the j'th class; the diagonal contains the
    # responses to the home-class.
    mu = np.full((nEn,nEn), np.nan)
    sig = np.full((nEn,nEn), np.nan)
    for i,resp in enumerate(results):
        mu[i,:] = resp['post_mean_resp']
        sig[i,:] = resp['post_std_resp']

    # for each EN:
    # get the likelihood of each puff (ie each col of post_train_resp)
    likelihoods = np.zeros((n_post,nEn))
    for i in range(n_post):

        dist = (np.tile(post_train_resp[:,i],(10,1)) - mu) / sig # 10 x 10 matrix
        # The ith row, jth col entry is the mahalanobis distance of this test
        # digit's response from the i'th ENs response to the j'th class.
        # For example, the diagonal contains the mahalanobis distance of this
        # digit's response to each EN's home-class response.

        # 1. Apply rewards for above-threshold responses:
        off_diag = dist - np.diag(np.diag(dist))
        on_diag = np.diag(dist).copy()
        # Reward any onDiags that are above some threshold (mu - n*sigma) of an EN.
        # CAUTION: This reward-by-shrinking only works when off-diagonals are
        # demolished by very high value of 'home_advantage'.
        home_threshs = home_thresh_sigmas * np.diag(sig)
        # aboveThreshInds = np.nonzero(on_diag > home_threshs)[0]
        on_diag[on_diag > home_threshs] /= above_home_thresh_reward
        on_diag = np.diag(on_diag) # turn back into a matrix
        # 2. Emphasize the home-class results by shrinking off-diagonal values.
        # This makes the off-diagonals less important in the final likelihood sum.
        # This is shrinkage for a different purpose than in the lines above.
        dist = (off_diag / home_advantage) + on_diag
        likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener
        # In pure thresholding case (ie off-diagonals ~ 0), this does not matter.

    # make predictions:
    pred_classes = np.argmin(likelihoods, axis=1)
    # for i in range(n_post):
        # pred_classes[i] = find(likelihoods(i,:) == min(likelihoods(i,:) ) )

    # calc accuracy percentages:
    class_acc = np.zeros(nEn)
    for i in range(nEn):
        class_acc[i] = (100*np.logical_and(pred_classes == i, true_classes == i).sum())/(true_classes == i).sum()

    total_acc = (100*(pred_classes == true_classes).sum())/len(true_classes)

    # confusion matrix:
    # i,j'th entry is number of test digits with true label i that were predicted to be j.
    confusion = confusion_matrix(true_classes, pred_classes)

    # measure ROC AUC for each class
    roc_dict = roc_multi(true_classes, likelihoods)

    output = dict() # initialize dictionary
    output['true_classes'] = true_classes
    output['targets'] = roc_dict['targets']
    output['roc_auc'] = roc_dict['roc_auc']
    output['fpr'] = roc_dict['fpr']
    output['tpr'] = roc_dict['tpr']
    output['pred_classes'] = pred_classes
    output['likelihoods'] = likelihoods
    output['acc_perc'] = class_acc
    output['total_acc'] = total_acc
    output['conf_mat'] = confusion
    output['home_advantage'] = home_advantage
    output['home_thresh_sigmas'] = home_thresh_sigmas
    output['above_home_thresh_reward'] = above_home_thresh_reward

    return output

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
