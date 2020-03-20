# Program to calculate rms error

# Library
import numpy as np

# RMSE function
def rms_error(x, y):   

    d = x - y
    d2 = d ** 2
    rmse_val = np.sqrt(d2.mean())
    return rmse_val