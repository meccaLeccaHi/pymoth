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
    3. The rest is simple calculation
    '''

    import pdb; pdb.set_trace()

    r = results
    # # nEn = len(r) # number of ENs, same as number of classes
    # ptInds = np.nonzero(r[2].postTrainOdorResp >= 0) # indices of post-train (ie validation) digits
    # # DEV NOTE: Why use 2 as index above?
    # # nP = length(ptInds) # number of post-train digits
    #
    # # extract true classes:
    # temp = [r[1].odorClass] # throughout, digits may be referred to as odors or 'odor puffs'
    # trueClasses = temp[ptInds]
    #
    # # extract the relevant odor puffs: Each row is an EN, each col is an odor puff
    # for i = 1:len(r)
    #   temp = r[i].postTrainOdorResp
    #   ptResp[i,:] = temp(ptInds)
    # end
    #
    # # make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class:
    # for i = 1:len(r)
    #   mu[i,:] = r[i].postMeanResp
    #   sig[i,:] = r[i].postStdResp
    # end
    #
    # # for each EN:
    # # get the likelihood of each puff (ie each col of ptResp)
    # likelihoods = zeros(len(ptInds),len(r))
    # for i = 1:len(ptInds)
    #   dist = ( repmat(ptResp(:,i), [ 1, len(r) ]) - mu )./ sig # 10 x 10 matrix. The ith row, jth col entry is the mahalanobis distance of
    #                   # this test digit's response from the i'th ENs response to the j'th class. For example, the diagonal contains
    #                   # the mahalanobis distance of this digit's response to each EN's home-class response.
    #   likelihoods[i,:] = sum(dist.^4) # the ^4 (instead of ^2) is a sharpener
    # end
    #
    # # make predictions:
    # for i = 1:len(ptInds)
    #   predClasses[i] = find(likelihoods[i,:] == min(likelihoods[i,:] ) )
    # end
    #
    # # calc accuracy percentages:
    # for i = 1:len(r)
    #   classAccuracies[i] = 100*sum(predClasses == i & trueClasses == i)/sum(trueClasses == i)
    # end
    # totalAccuracy = 100*sum(predClasses == trueClasses)/length(trueClasses)
    #
    # # confusion matrix:
    # # i,j'th entry is number of test digits with true label i that were predicted to be j.
    # confusion = confusionmat(trueClasses, predClasses)
    #
    # output.trueClasses = trueClasses
    # output.predClasses = predClasses
    # output.likelihoods = likelihoods
    # output.accuracyPercentages = classAccuracies
    # output.totalAccuracy = totalAccuracy
    # output.confusionMatrix = confusion

    return output
