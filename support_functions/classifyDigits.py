def classifyDigitsViaLogLikelihood( results ):
    '''
    function [output] = classifyDigitsViaLogLikelihood_fn (results)
    Classify the test digits in a run using log likelihoods from the various EN responses:
    Inputs:
    results = 1 x 10 struct produced by viewENresponses. i'th entry gives results for all classes, in the i'th EN.
    Important fields:
    1. postMeanResp, postStdResp (to calculate post-training, ie val, digit response distributions).
    2. postTrainOdorResponse (gives the actual responses for each val digit, for that EN)
        Note that non-post-train odors have response = -1 as a flag.
    3. odorClass: gives the true labels of each digit, 0 to 9 (10 = '0'). this is the same in each EN.

    output:
    output = struct with the following fields:
    1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries are summed log likelihoods.
    2. trueClasses = shortened version of whichOdor (with only postTrain, ie validation, entries)
    3. predClasses = predicted classes
    4. confusionMatrix = raw counts, rows = ground truth, cols = predicted
    5. classAccuracies = 1 x 10 vector, with class accuracies as percentages
    6. totalAccuracy = overall accuracy as percentage

    ------------------------------------------------------------------------------

    plan:
    1. for each test digit (ignore non-postTrain digits), for each EN, calculate the # stds from the test
    digit is from each class distribution. This makes a 10 x 10 matrix
    where each row corresponds to an EN, and each column corresponds to a class.
    2. Square this matrix by entry. Sum the columns. Select the col with the lowest value as the predicted
    class. Return the vector of sums in 'likelihoods'.
    3. The rest is simple calculation.
    '''

    import numpy as np
    from sklearn.metrics import confusion_matrix # DEV NOTE: Should consider removing
        # This is the only thing we are using sklearn for

    # r = results
    nEn = len(results) # number of ENs, same as number of classes
    ptInds = np.nonzero(results[1]['postTrainOdorResp'] >= 0)[0] # indices of post-train (ie validation) digits
    # DEV NOTE: Why use 2 (1, here) as index above?
    nP = len(ptInds) # number of post-train digits

    # extract true classes:
    trueClasses = results[0]['odorClass'][ptInds] # throughout, digits may be referred to as odors or 'odor puffs'
    # DEV NOTE: Why use 1 (0, here) as index above?

    # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    ptResp = np.full((nEn,nP), np.nan)
    for i,resp in enumerate(results):
        ptResp[i,:] = resp['postTrainOdorResp'][ptInds]

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class:
    mu = np.full((nEn,nEn), np.nan)
    sig = np.full((nEn,nEn), np.nan)
    for i,resp in enumerate(results):
        mu[i,:] = resp['postMeanResp']
        sig[i,:] = resp['postStdResp']

    # for each EN:
    # get the likelihood of each puff (ie each col of ptResp)
    likelihoods = np.zeros((nP,nEn))
    for i in range(nP):
        # Caution: ptResp[:,i] becomes a row vector, but we need it to stay as a
        # col vector so we can make 10 identical columns. So transpose it back with [np.newaxis]
        a = ptResp[:,i][np.newaxis]
        dist = ( np.tile( a.T, ( 1, 10 )) - mu) / sig # 10 x 10 matrix
        # The ith row, jth col entry is the mahalanobis distance of this test
        # digit's response from the i'th ENs response to the j'th class.
        # For example, the diagonal contains the mahalanobis distance of this
        # digit's response to each EN's home-class response.

        likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener

    # make predictions:
    predClasses = np.argmin(likelihoods, axis=1)

    # calc accuracy percentages:
    classAccuracies = np.zeros(nEn)
    for i in range(nEn):
        classAccuracies[i] = (100*np.logical_and(predClasses == i, trueClasses == i).sum())/(trueClasses == i).sum()

    totalAccuracy = (100*(predClasses == trueClasses).sum())/len(trueClasses)

    # confusion matrix:
    # i,j'th entry is number of test digits with true label i that were predicted to be j.
    confusion = confusion_matrix(trueClasses, predClasses)

    # DEV NOTE: could assign these directly above and save this step.
    output = dict()
    output['trueClasses'] = trueClasses
    output['predClasses'] = predClasses
    output['likelihoods'] = likelihoods
    output['accuracyPercentages'] = classAccuracies
    output['totalAccuracy'] = totalAccuracy
    output['confusionMatrix'] = confusion

    return output


def classifyDigitsViaThresholding(results, homeAdvantage, homeThresholdSigmas, aboveHomeThreshReward):
    '''
    Classify the test digits in a run using log likelihoods from the various EN
    responses, with the added option of rewarding high scores relative to an ENs
    home-class expected response distribution.
    One use of this function is to apply de-facto thresholding on discrete ENs,
    so that the predicted class corresponds to the EN that spiked most strongly
    (relative to its usual home-class response).
    Inputs:
     1. results = 1 x 10 struct produced by viewENresponses. i'th entry gives
     results for all classes, in the i'th EN.
      Important fields:
        a. postMeanResp, postStdResp (to calculate post-training, ie val, digit
        response distributions).
        b. postTrainOdorResponse (gives the actual responses for each val digit,
        for that EN)
            Note that non-post-train odors have response = -1 as a flag.
        c. odorClass: gives the true labels of each digit, 1 to 10 (10 = '0').
        This is the same in each EN.
     2. 'homeAdvantage' is the emphasis given to the home EN. It
            multiplies the off-diagonal of dist. 1 -> no advantage (default).
            Very high means that a test digit will be classified according to
            the home EN it does best in, ie each EN acts on it's own.
     3. 'homeThresholdSigmas' = the number of stds below an EN's home-class mean
            that we set a threshold, such that if a digit scores above this threshold
            in an EN, that EN will be rewarded by 'aboveHomeThreshReward' (below)
     4. 'aboveHomeThreshReward': if a digit's response scores above the EN's mean
            home-class value, reward it by dividing by aboveHomeThreshReward. This
            reduces the log likelihood score for that EN.

    Output:
       A dictionary with the following fields:
       1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries
       are summed log likelihoods.
       2. trueClasses = shortened version of whichOdor (with only postTrain, ie
       validation, entries)
       3. predClasses = predicted classes
       4. confusionMatrix = raw counts, rows = ground truth, cols = predicted
       5. classAccuracies = 1 x 10 vector, with class accuracies as percentages
       6. totalAccuracy = overall accuracy as percentage

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
        homeAdvantage = 1
        homeThresholdSigmas = any number
        aboveHomeThreshReward = 1
    The following value enables pure home-class thresholding:
        homeAdvantage = 1e12 # to effectively eliminate off-diagonals
    '''

    import numpy as np
    from sklearn.metrics import confusion_matrix

    # r = results
    nEn = len(results) # number of ENs, same as number of classes
    ptInds = np.nonzero(results[1]['postTrainOdorResp'] >= 0)[0] # indices of post-train (ie validation) digits
    # DEV NOTE: Why use 2 (1, in Python) as index above?
    nP = len(ptInds) # number of post-train digits

    # extract true classes:
    trueClasses = results[0]['odorClass'][ptInds] # throughout, digits may be referred to as odors or 'odor puffs'
    # DEV NOTE: Why use 1 (0, in Python) as index above?

    # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    ptResp = np.full((nEn,nP), np.nan)
    for i,resp in enumerate(results):
        ptResp[i,:] = resp['postTrainOdorResp'][ptInds]

    # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class.
    # For example, the i'th row, j'th col entry of 'mu' is the mean of the i'th
    # EN in response to digits from the j'th class; the diagonal contains the
    # responses to the home-class.
    mu = np.full((nEn,nEn), np.nan)
    sig = np.full((nEn,nEn), np.nan)
    for i,resp in enumerate(results):
        mu[i,:] = resp['postMeanResp']
        sig[i,:] = resp['postStdResp']

    # for each EN:
    # get the likelihood of each puff (ie each col of ptResp)
    likelihoods = np.zeros((nP,nEn))
    for i in range(nP):

        dist = (np.tile(ptResp[:,i],(10,1)) - mu) / sig # 10 x 10 matrix
        # The ith row, jth col entry is the mahalanobis distance of this test
        # digit's response from the i'th ENs response to the j'th class.
        # For example, the diagonal contains the mahalanobis distance of this
        # digit's response to each EN's home-class response.

        # 1. Apply rewards for above-threshold responses:
        offDiag = dist - np.diag(np.diag(dist))

        ## DEV NOTE: DO WE NEED THE .copy() BELOW?
        onDiag = np.diag(dist).copy()
        # Reward any onDiags that are above some threshold (mu - n*sigma) of an EN.
        # CAUTION: This reward-by-shrinking only works when off-diagonals are
        # demolished by very high value of 'homeAdvantage'.
        homeThreshs = homeThresholdSigmas * np.diag(sig)
        aboveThreshInds = np.nonzero(onDiag > homeThreshs)[0]
        onDiag[onDiag > homeThreshs] /= aboveHomeThreshReward
        onDiag = np.diag(onDiag) # turn back into a matrix
        # 2. Emphasize the home-class results by shrinking off-diagonal values.
        # This makes the off-diagonals less important in the final likelihood sum.
        # This is shrinkage for a different purpose than in the lines above.
        dist = (offDiag / homeAdvantage) + onDiag
        likelihoods[i,:] = np.sum(dist**4, axis=0) # the ^4 (instead of ^2) is a sharpener
        # In pure thresholding case (ie off-diagonals ~ 0), this does not matter.

    # make predictions:
    predClasses = np.argmin(likelihoods, axis=1)
    # for i in range(nP):
        # predClasses[i] = find(likelihoods(i,:) == min(likelihoods(i,:) ) )

    # calc accuracy percentages:
    classAccuracies = np.zeros(nEn)
    for i in range(nEn):
        classAccuracies[i] = (100*np.logical_and(predClasses == i, trueClasses == i).sum())/(trueClasses == i).sum()

    totalAccuracy = (100*(predClasses == trueClasses).sum())/len(trueClasses)

    # confusion matrix:
    # i,j'th entry is number of test digits with true label i that were predicted to be j.
    confusion = confusion_matrix(trueClasses, predClasses)

    # DEV NOTE: could assign these directly above and save this step.
    output = dict()
    output['homeAdvantage'] = homeAdvantage
    output['trueClasses'] = trueClasses
    output['predClasses'] = predClasses
    output['likelihoods'] = likelihoods
    output['accuracyPercentages'] = classAccuracies
    output['totalAccuracy'] = totalAccuracy
    output['confusionMatrix'] = confusion
    output['homeAdvantage'] = homeAdvantage
    output['homeThresholdSigmas'] = homeThresholdSigmas
    output['aboveHomeThreshReward'] = aboveHomeThreshReward

    return output
