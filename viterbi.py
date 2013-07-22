from lmatrix import LMatrix
from collections import defaultdict
import numpy as np

def viterbi(obs, A, B, pi):
    """
    given the observations and the HMM setup
    
    return the most likely state sequence, the prob trellis, the backtrace pointers
    """
    Q = A.rlabels
    T = len(obs)

    #trellis[s,i] means:
    #The probability of **the most probable path** ending in state s with observation obs[i]
    trellis = LMatrix(Q, xrange(T))
    
    backtrace = [dict()] * T
    
    for i in xrange(T):
        ob = obs[i]
        for s in Q:
            if i == 0:
                print s, i, np.log2(B[s,ob]), np.log2(pi[s])
                trellis[s,i] = np.log2(B[s,ob]) + np.log2(pi[s])
            else:
                probs = trellis[:,i-1]  + np.log2(A[:,s])
                trellis[s,i] = np.log2(B[s,ob]) + np.max(probs)

                backtrace[i][s] = Q[np.argmax(probs)]

    print ("", ), trellis, backtrace