import numpy as np

def wrap_angle(val):
    return (val + np.pi) % (2 * np.pi) - np.pi

def calc_cum_sum(input_arr, mask_array):
    # TODO: fix cumululative sum with mask
    return input_arr
