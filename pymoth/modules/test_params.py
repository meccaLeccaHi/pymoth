#!/usr/bin/env python3

# import packages and modules
import numpy as np
from .params import ModelParams, ExpParams

def main():

    print('Testing params module:')

    # test ModelParams( active_pixel_inds, goal )
    model_params = ModelParams( 10, 10 )
    print('\tModelParams class test passed')

    # test ModelParams.create_connection_matrix()
    model_params.create_connection_matrix()
    print('\tcreate_connection_matrix method test passed')

    # test ExpParams(train_classes, class_labels, val_per_class )
    experiment_params =  ExpParams( np.array(range(10)), np.array(range(10)), 1 )
    print('\tExpParams class test passed')

if __name__ == '__main__':
    main()
