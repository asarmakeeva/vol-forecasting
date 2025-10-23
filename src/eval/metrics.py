import numpy as np


def qlike(y, yhat):
    eps = 1e-8
    return np.mean(np.log(yhat**2 + eps) + (y**2)/(yhat**2 + eps))