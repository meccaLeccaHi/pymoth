#!/usr/bin/env python3

# import packages and modules
import numpy as np
from .sde import sde_wrap
from .params import ModelParams, ExpParams

def main():

    print('Testing params module:')

    # create dummy data
    dummy_model_params = ModelParams( 10, 10 )
    dummy_model_params.create_connection_matrix()
    dummy_exp_params =  ExpParams( np.array(range(10)), np.array(range(10)), 1 )
    import pdb; pdb.set_trace()

    # test sde_wrap
    sde_wrap( dummy_model_params, dummy_exp_params, dummy_feature_array )
    print('\tsde_wrap method test passed')

    # test sde_evo_mnist
    # sde_evo_mnist(tspan, init_cond, time, class_mag_mat, feature_array,
    #     octo_hits, mP, exP, seed_val)
