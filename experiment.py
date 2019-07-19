#!/usr/bin/env python3

from moth_net import MothNet

# Instantiate the MothNet object
mothra = MothNet()

# call our instance methods
digit_queues = mothra.load_MNIST()
train_X, val_X, train_y, val_y = mothra.train_test_split(digit_queues)

# Load parameters
moth_parameters = mothra.load_moth() # define moth model parameters
experiment_parameters = mothra.load_exp() # define parameters of a time-evolution experiment

# Run simulation (SDE time-step evolution)
sim_results = mothra.simulate(moth_parameters, experiment_parameters, digit_queues)
# mothra.fit(X_train, y_train)

print(sim_results)

# mnist_accuracy = mothra.score_on_MNIST()
# svm_accuracy = mothra.score_svm(X_test, y_test)
# knn_accuracy = mothra.score_knn(X_test, y_test)
