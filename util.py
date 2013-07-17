import numpy as np

def almost_eq(this, that, decimal = 5):
    return np.all((this - that) < 10 ** (-decimal))