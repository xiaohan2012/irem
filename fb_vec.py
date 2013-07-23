from lmatrix import LMatrix
import numpy as np

def forward_vec(obs, A, B, pi):
    Q = A.rlabels
    V = B.clabels
    T = len(obs)
    
    tbl = LMatrix(Q,xrange(T))

    A_mat = np.matrix(A)
    
    for i in xrange(T):
        ob = obs[i]
        diag = np.diag(B[:,ob])
        tbl[:,i] = np.matrix(tbl[:,i-1]) * A_mat * diag if i > 0 else pi * diag 
        tbl[:,i] = tbl[:,i] / sum(tbl[:,i])
        
    obs_prob = 0
    return tbl, obs_prob