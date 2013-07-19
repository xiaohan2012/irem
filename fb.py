import numpy as np

from lmatrix import LMatrix

from util import memoized

@memoized
def forward_prob_table(obs, A, B, pi):
    """
    (
    obs, the observations: list of obj,
    A, State transition matrix: LMatrix,
    B, state emission matrix: LMatrix
    pi, initial state vector: tuple of tuple)
    =>
    forward prob table: LMatrix
    """
    states = A.rlabels
    T = len(obs)
    
    #initialize forward prob table
    #ft[s,j] means the probability of seeing the first `j` observations
    #and the `j`th state is `s`
    ft = LMatrix(rlabels = states, #states as row
                 clabels = range(T)) #observations as columns

    
    for s in states:
        #somewhat not easy to read, because pi must be hashable, so the original dict becomes a tuple
        #my question: any hashable dict?
        ft[s, 0] = [i for i in pi if i[0] == s][0][1] * B[s,obs[0]]

    for i in xrange(1, T):
        for s in states:
            ft[s,i] = sum(ft[:,i-1] * A[:, s] * B[s, obs[i]])
    """
    return LMatrix(rlabels = states + ["Total"],
                   clabels = ft.clabels,
                   data = np.array(ft.tolist() + [ft.sum(0).tolist()])
               )
    """
    return ft
    
@memoized
def backward_prob_table(obs, A, B, pi):
    states = A.rlabels

    T = len(obs)
    
    #forward prob table
    #ft[s,j] means the prob of observing the `j+1`-to-end's observations and at time `j`, in state `s`
    ft = LMatrix(rlabels = states, #states as row
                 clabels = range(T)) #observations as columns

    for s in states:
        ft[s, -1] = 1

    for i in range(T - 1)[::-1]:
        for s in states:
            ft[s,i] = sum(A[s,:] * ft[:,i+1] * B[:,obs[i+1]])

    """
    obs_prob = np.zeros(len(ft.clabels))
    obs_prob[1:] = ft[:,1:].sum(0)
    obs_prob[0] = sum(ft[:,0] * np.array([tpl[1] for tpl in pi]))
    
    return LMatrix(rlabels = states + ["Total"],
                   clabels = ft.clabels,
                   data = np.array(ft.tolist() + [obs_prob])
               )
    """
    return ft
            
