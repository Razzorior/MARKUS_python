# general helper functions

import os
import numpy as np

def makedirs(path_list):
    if type(path_list) == list:
        for path in path_list:
            if not os.path.isdir(path):
                os.makedirs(path)
    else:
        raise ValueError("This makedirs() function expects a list!")


def tanh(x, inv=False):
    tan = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if inv:
        tan = -tan
    return (tan + 1) / 2.

def zero_one_feature_scaling(nd_array):
    # scale array to be normalized between 0,1 in every column
    nd_array = nd_array - np.min(nd_array, 0)
    nd_array = nd_array / np.max(nd_array, 0)
    return nd_array