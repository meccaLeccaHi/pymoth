#!/usr/bin/env python3

import numpy as _np

from classify import roc_multi

# test _ROC_multi
roc_multi(_np.array([1,2,3]),_np.array([[1,2,3],[1,2,3],[1,2,3]]))
