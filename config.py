INAOE_DATASET_DIR = '/home/abian/Data/Dataset/GTMA/DiabeticFootDataset/INAOE/Data/'

import numpy as np
def sliding_window(a, window=2):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

initial_temperature = 18 # Temperature lower bound
EXPECTED_VALUE_INTERVAL = np.cumsum([0, 4, 4, 1, 1, 1, 1, 1, 1, 1, 4]) + initial_temperature
EXPECTED_VALUE_CLUSTER = sliding_window(EXPECTED_VALUE_INTERVAL, 2).mean(axis=1)