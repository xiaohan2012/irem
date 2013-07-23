import unittest
import sys

import numpy as np

sys.path.append("..")

from lmatrix import LMatrix
from util import *

def weather_data():
    pi = np.matrix([0.5, 0.5])
        
    A = LMatrix(rlabels = ["rain", "sunny"],
                data = np.array([
                    [0.7, 0.3],
                    [0.3, 0.7]
                ]))
        
    B = LMatrix(rlabels = ["rain", "sunny"],
                clabels = ["umbrella", "no-umbrella"],
                data = np.array([
                    [0.9, 0.1],
                    [0.2, 0.8],
                ]))

    obs = ("umbrella", "umbrella", "no-umbrella", "umbrella", "umbrella")

    return obs, A, B, pi
