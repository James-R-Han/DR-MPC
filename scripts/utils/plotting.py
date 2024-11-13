import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w