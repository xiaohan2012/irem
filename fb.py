import numpy as np

from lmatrix import LMatrix

from util import memoized

START = "START"
END = "END"

@memoized
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

@memoized
def backward_prob_table(obs, A, B):
    states = A.rlabels
    
    #forward prob table
    ft = LMatrix(rlabels = states, #states as row
                 clabels = range(len(obs))) #observations as columns

    for s in states:
        ft[s, -1] = A[s, END]

    for i in range(len(obs) - 1)[::-1]:
        for s in states:
            ft[s,i] = sum(ft[:,i+1] * A[s,:] * B[:,obs[i+1]])

    return sum(ft[:,0] * A[START,:] * B[:, obs[0]]),ft


def convergent(old_mat, new_mat):
    """whether two matrices are approximate enough"""
    print (np.abs(old_mat - new_mat) / old_mat)
    return (np.abs(old_mat - new_mat) / old_mat < 1e-5).all()
    
def forward_backward(lst_of_obs, V, Q, init_A, init_B):
    """
    (
    lst_of_obs: list of tuple of str, list of observation sequence,
    V: set of str, output vocabulary,
    Q: set of str hidden states set,
    A: initial transition prob matrix,
    B: initial emission prob matrix
    ) =>
    (
    A: transition prob matrix
    B: emission prob matrix
    )

    the backward-forward algorithm
    """
    #observation count
    m = len(lst_of_obs)
    
    #initial A and B
    A = init_A
    B = init_B
    
    while True:
        #expect
        @memoized
        def gamma(k, t, j):
            """the kth observation, at time t and for the jth hidden state"""
            obs = lst_of_obs[k]
            T = len(obs) #obs sequence length
            
            ft = forward_prob_table(obs, A, B)
            obs_prob, bt = backward_prob_table(obs, A, B)
            return ft[j,t] * bt[j,t] / obs_prob
                
        @memoized
        def xi(k, t, i, j):
            """the kth observation, at time t and from state i to state j"""
            
            obs = lst_of_obs[k]
            T = len(obs) #obs sequence length
            
            ft = forward_prob_table(obs, A, B)
            obs_prob, bt = backward_prob_table(obs, A, B)
            return ft[j,t] * A[i,j] * B[j,obs[t+1]] * bt[j,t+1] / obs_prob

        #maximization
        new_A = LMatrix(rlabels = Q)
        new_B = LMatrix(rlabels = Q, clabels = V)

        for i in Q:
            for j in Q:
                new_A[i,j] = sum((xi(k, t, i, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]) - 1))) / \
                             sum((xi(k, t, i, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]) - 1) for j in Q ))

        for j in Q:
            for vk in V:
                new_B[j,vk] = sum((gamma(k, t, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k])) if lst_of_obs[k][t] == vk)) / \
                              sum((gamma(k, t, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]))))
        print new_A
        print new_B
        if convergent(A, new_A) and convergent(B, new_B):
            return new_A, new_B         