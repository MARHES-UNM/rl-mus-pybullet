import numpy as np

def wrap_angle(val):
    return (val + np.pi) % (2 * np.pi) - np.pi
