import numpy as np


def get_rms(records):
    """
    """
    return np.math.sqrt(sum([x ** 2 for x in records]) / len(records))