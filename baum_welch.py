import numpy as np

from util import memoized

def convergent(old_mat, new_mat):
    """whether two matrices are approximate enough"""
    return (np.abs(old_mat - new_mat) / old_mat < 1e-5).all()

@memoized    
def gamma(obs, j, from_state, to_state, A, B, pi):
    """
    (observation: list of str, time j: int, from state: str, to state: str) => float
    
    for observation `obs`, the probability that has `from_state` as its `j`th state and `to_state` as its (`j`+1)th state
    """
    from fb import forward_prob_table, backward_prob_table
    
    ft, obs_prob1 = forward_prob_table(obs, A, B, pi)
    bt, obs_prob2 = backward_prob_table(obs, A, B, pi)

    #ensure obs_prob from both tables are the same
    assert(np.abs(obs_prob1 - obs_prob2) < 1e-5)
    #print ft[from_state, j] , A[from_state, to_state] , B[to_state, obs[j+1]] , bt[to_state, j+1]
    #print obs_prob1
    return ft[from_state, j] * A[from_state, to_state] * B[to_state, obs[j+1]] * bt[to_state, j+1] / obs_prob1

@memoized
def delta(obs, j, s, A, B, pi):
    pass
    
def baum_welch(lst_of_obs, V, Q, init_A, init_B):
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
        print "starting A "
        print A
        
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
        new_A = LMatrix(rlabels = A.rlabels)
        new_B = LMatrix(rlabels = A.rlabels, clabels = V)

        for i in Q:
            for j in Q:
                numer = sum((xi(k, t, i, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]) - 1)))
                denom = sum((xi(k, t, i, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]) - 1) for j in Q ))
                #print i, j, numer, denom
                new_A[i,j] =  numer / denom
                              

        for j in Q:
            for vk in V:
                #print j, vk, numer, denom
                numer = sum((gamma(k, t, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k])) if lst_of_obs[k][t] == vk))
                denom = sum((gamma(k, t, j) for k in xrange(m) for t in xrange(len(lst_of_obs[k]))))
                new_B[j,vk] = numer / denom
                
        print "new A"
        print new_A
        #print new_B

        sliced_A = A[1:-1,1:-1]
        sliced_new_A = new_A[1:-1,1:-1]
        sliced_B = B[1:-1,1:-1]
        sliced_new_B = new_B[1:-1,1:-1]
        
        if convergent(sliced_A, sliced_new_A) and convergent(B, new_B):
            return new_A, new_B
        break
        A = new_A
        B = new_B
        print np.sum(sliced_A - sliced_new_A)