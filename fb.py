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
    return (np.abs(old_mat - new_mat) / old_mat < 1e-5).all()
    
def forward_backward(lst_of_obs, V, Q):
    """
    (
    lst_of_obs: list of list of str, list of observation sequence,
    V: set of str, output vocabulary,
    Q: set of str hidden states set,
    ) =>
    (
    A: transition prob matrix
    B: emission prob matrix
    )

    the backward-forward algorithm
    """
    #init guess of A and B
    A = LMatrix(rlabels = V, data = np.random.randn(len(V), len(V)))
    B = LMatrix(rlabels = V, clabels = Q, data = np.random.ramdn(len(V), len(Q)))

    while True:
        for obs in lst_of_obs:
            T = len(obs) #obs sequence length

            ft = forward_prob_table(obs, A, B)
            obs_prob, bt = backward_prob_table(obs, A, B)

            #expect
            @memoized
            def gamma(t,j):
                return ft[j,t] * bt[j,t] / obs_prob
                
            @memoized
            def xi(t,i,j):
                return ft[j,t] * A[i,j] * B[j,obs[t+1]] * bt[j,t+1] / obs_prob

            #maximization
            new_A = LMatrix(rlabels = V)
            new_B = LMatrix(rlabels = V, clabels = Q)

            for i in Q:
                for j in Q:
                    new_A[r,c] = sum((xi(t,i,j) for t in xrange(T - 1))) / \
                                 sum((xi(t,i,j) for t in xrange(T - 1) for j in new_A.rlabels))

            for j in Q:
                for vk in V:
                    new_B[j,vk] = sum((gamma(t,j) for t in xrange(T) if obs[t] == vk)) / \
                                  sum((gamma(t,j) for t in xrange(T)))

            A = new_A
            B = new_B


        if convergent:
            break
            
    return A,B