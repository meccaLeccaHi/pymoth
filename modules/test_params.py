#!/usr/bin/env python3

# import packages and modules
import numpy as np
from params import ModelParams, ExpParams

# test ModelParams( active_pixel_inds, goal )
model_params = ModelParams( 10, 10 )

# test ModelParams.create_connection_matrix()
model_params.create_connection_matrix()

# test ExpParams(train_classes, class_labels, val_per_class )
experiment_params =  ExpParams( np.array(range(10)), np.array(range(10)), 1 )
