#!/usr/bin/env python3
from moth_net import MothNet

def main():

    # instantiate the MothNet object
    mothra = MothNet()

    # loop through the number of simulations specified:
    for run in range(mothra.NUM_RUNS):

        # generate dataset
        digit_queues = mothra.load_MNIST()
        train_X, val_X, train_y, val_y = mothra.train_test_split(digit_queues)

        # load parameters
        moth_parameters = mothra.load_moth() # define moth model parameters
        experiment_parameters = mothra.load_exp() # define parameters of a time-evolution experiment

        # run simulation (SDE time-step evolution)
        sim_results = mothra.simulate(moth_parameters, experiment_parameters, digit_queues)
        # future: mothra.fit(X_train, y_train)

        # collect response statistics:
        # process the sim results to group EN responses by class and time
        EN_resp_trained = mothra.collect_stats(sim_results, experiment_parameters,
            mothra.class_labels, mothra.SHOW_TIME_PLOTS, mothra.SHOW_ACC_PLOTS,
            images_filename=mothra.RESULTS_FILENAME, images_folder=mothra.RESULTS_FOLDER,
            screen_size=mothra.SCREEN_SIZE)

        # reveal scores
        # print scores for MothNet
        mothra.score_moth_on_MNIST(EN_resp_trained)
        # # print scores for KNN
        # mothra.score_knn(X_test, y_test)
        # # print scores for SVM
        # mothra.score_svm(X_test, y_test)

if __name__ == "__main__":
    main()
