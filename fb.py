import numpy as np

from lmatrix import LMatrix

START = "START"
END = "END"

def forward_prob_table(obs, A, B):
    """
    (observations: list of obj,
    State transition matrix: LMatrix,
    state emission matrix: LMatrix) =>
    forward prob table: LMatrix
    """
    states = A.rlabels
    #forward prob table
    ft = LMatrix(rlabels = states, #states as row
                 clabels = range(len(obs))) #observations as columns

    
    for s in states:
        ft[s, 0] = A[START,s] * B[s, obs[0]]

    for prev_i, i in zip(range(len(obs) -1 ), range(1, len(obs))):
        ob = obs[i]
        for s in states:
            ft[s,i] = sum(ft[:,prev_i] * A[:, s] * B[s, ob])
    
    ft[END, -1] = sum(ft[:, -1] * A[:, END])
    
    return ft