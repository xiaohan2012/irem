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
    
    backtrace = [dict() for i in xrange(T-1)]
    
    for i in xrange(T):
        ob = obs[i]
        for s in Q:
            if i == 0:
                trellis[s,i] = np.log2(B[s,ob]) + np.log2(pi[s])
            else:
                probs = trellis[:,i-1]  + np.log2(A[:,s])
                trellis[s,i] = np.log2(B[s,ob]) + np.max(probs)
                
                backtrace[i-1][s] = Q[np.argmax(probs)]
                
    most_likely_end_state = Q[np.argmax(trellis[:,-1])]
    
    states = [most_likely_end_state]

    for cell in backtrace[::-1]:
        states.append(cell[states[-1]])
    
    return tuple(states[::-1]), trellis, backtrace


if __name__ == '__main__':
    from util import load_HMM
    A, B, pi = load_HMM()