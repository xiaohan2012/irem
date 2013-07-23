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
        
    return tbl


def backward_vec(obs, A, B, pi):
    Q = A.rlabels
    V = B.clabels
    T = len(obs)
    
    tbl = LMatrix(Q,xrange(T))

    A_mat = np.matrix(A)

    for i in range(T)[::-1]:
        ob = obs[i]
        O = np.diag(B[:,ob])

        r = np.array(A_mat * O * np.matrix(tbl[:,i+1]).T) \
                if i < T-1 else \
                    np.array(A_mat * O * np.ones((len(Q),1)))

        tbl[:,i] = r[:,0]
        
        tbl[:,i] = tbl[:,i] / sum(tbl[:,i])
            
    return tbl
    