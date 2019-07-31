#!/usr/bin/env python3

import pymoth
# from moth_net import MothNet
import time
import os

def main():

    runStart = time.time() # time execution duration

    # instantiate the MothNet object
    mothra = pymoth.MothNet()

    # loop through the number of simulations specified:
    for run in range(mothra.NUM_RUNS):

        # generate dataset
        digit_queues = mothra.load_MNIST()
        train_X, test_X, train_y, test_y = mothra.train_test_split(digit_queues)

        # load parameters
        mothra.load_moth() # define moth model parameters
        mothra.load_exp() # define parameters of a time-evolution experiment

        # run simulation (SDE time-step evolution)
        sim_results = mothra.simulate(digit_queues)
        # future: mothra.fit(X_train, y_train)

        # collect response statistics:
        # process the sim results to group EN responses by class and time
        EN_resp_trained = mothra.collect_stats(sim_results, mothra.experiment_params,
            mothra._class_labels, mothra.SHOW_TIME_PLOTS, mothra.SHOW_ACC_PLOTS,
            images_filename=mothra.RESULTS_FILENAME, images_folder=mothra.RESULTS_FOLDER,
            screen_size=mothra.SCREEN_SIZE)

        # reveal scores
        # score MothNet
        mothra.score_moth_on_MNIST(EN_resp_trained)
        # score KNN
        mothra.score_knn(train_X, train_y, test_X, test_y)
        # score SVM
        mothra.score_svm(train_X, train_y, test_X, test_y)

        if mothra.SHOW_ROC_PLOTS:
            mothra.show_multi_roc(['MothNet', 'SVM', 'KNN'], mothra._class_labels,
            images_filename=mothra.RESULTS_FOLDER + os.sep + mothra.RESULTS_FILENAME + '_ROC_multi')

    runDuration = time.time() - runStart
    print('{} executed in {:.3f} minutes'.format(__file__, runDuration/60))
    print()
    print('         -------------All done-------------         ')

if __name__ == "__main__":
    main()
