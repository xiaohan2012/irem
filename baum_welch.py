import numpy as np

from util import memoized, hashdict

from fb import forward_prob_table, backward_prob_table

from lmatrix import LMatrix

def convergent(old_score, new_score, places = 4):
    """whether scores are approximate enough"""
    
    return np.abs((old_score - new_score) / old_score) < 10**(-places)

@memoized    
def gamma(obs, j, from_state, to_state, A, B, pi):
    """
    (observation: list of str, time j: int, from state: str, to state: str, transition matrix: LMatrix, emission matrix: LMatrix, hashdict: init state vector) => float
    
    for observation `obs`, the probability that has `from_state` as its `j`th state and `to_state` as its (`j`+1)th state
    """
    
    ft, obs_prob1 = forward_prob_table(obs, A, B, pi)
    bt, obs_prob2 = backward_prob_table(obs, A, B, pi)

    #ensure obs_prob from both tables are the same
    assert(np.abs(obs_prob1 - obs_prob2) < 1e-5)

    return ft[from_state, j] * A[from_state, to_state] * B[to_state, obs[j+1]] * bt[to_state, j+1] / obs_prob1

@memoized
def delta(obs, j, s, A, B, pi):
    """
    (observation: list of str, time j: int, from state: str, to state: str, transition matrix: LMatrix, emission matrix: LMatrix, hashdict: init state vector) => float
    
    for observation `obs`, the probability that its `j`th state is `s`, given A, B and pi
    """
    states = A.rlabels
    T = len(obs)
    
    if j < T - 1: #not the special case
        return sum( (gamma(obs, j, s, to_state, A, B, pi) for to_state in states) )
    else:
        ft, obs_prob = forward_prob_table(obs, A, B, pi)
        return ft[s, T-1] / obs_prob

def one_iter(lst_of_obs, A, B, pi):
    """
    given list of observations and the configuratin of HMM(A, B and pi),
    first expect, then maximize, last return a new HMM
    """
    Q = A.rlabels
    V = B.clabels
    
    #get pi
    pi_unnormalized = np.array(map(lambda s: sum((delta(obs, 0, s, A, B, pi) for obs in lst_of_obs)), Q))

    #normalize it
    pi_normalized = pi_unnormalized / np.sum(pi_unnormalized)
    
    #to hashdict
    pi_normalized = hashdict(zip(Q, pi_normalized))
    
    #get transition prob matrix
    A_unnormalized = LMatrix(rlabels = Q, clabels = Q)

    for obs in lst_of_obs:
        T = len(obs)
        for fs in Q:
            for ts in Q:
                A_unnormalized[fs, ts] += sum( (gamma(obs, j, fs, ts, A, B, pi) for j in xrange(T-1)) )

    #normalize it
    rc,cc =  A_unnormalized.shape
    A_normalized = A_unnormalized / A_unnormalized.sum(1).reshape(rc,1).repeat(cc,1)
    
    #get emission prob matrix
    B_unnormalized = LMatrix(rlabels = Q, clabels = V)

    for obs in lst_of_obs:
        for j, ob in enumerate(obs):
            for s in Q:
                B_unnormalized[s,ob] += delta(obs, j, s, A, B, pi)

    
    #normalize it
    rc,cc =  B_unnormalized.shape
    B_normalized = B_unnormalized / B_unnormalized.sum(1).reshape(rc,1).repeat(cc,1)
                
    return A_normalized, B_normalized, pi_normalized

def clear_memoization():
    gamma.cache = {}
    delta.cache = {}
    forward_prob_table.cache  = {}
    backward_prob_table.cache = {}

def take_snapshot(iteration, A, B, pi):
    #save a snap shot of the parameters

    from cPickle import dump, load
    dump(A.rlabels, open("param_snapshot/Q.vec", "w"))
    dump(B.clabels, open("param_snapshot/V.vec", "w"))

    dump(A,open("param_snapshot/%d_A.mat" %iteration, "w"))
    dump(B,open("param_snapshot/%d_B.mat" %iteration, "w"))
    dump(pi,open("param_snapshot/%d_pi.mat" %iteration, "w"))

    
    
def baum_welch(lst_of_obs, A, B, pi):
    """
    lst_of_obs: list of tuple of str, list of observation sequence,
    (
    A: initial transition prob matrix,
    B: initial emission prob matrix
    pi: initial state vector
    ) =>
    (
    A: transition prob matrix
    B: emission prob matrix,
    pi: initial state vector
    )

    the baum-welch algorithm
    """
    scores = []
    iteration = 0
    
    while True:
        take_snapshot(iteration, A, B, pi)
        
        old_score = sum( (np.log(forward_prob_table(obs, A, B, pi)[1]) for obs in lst_of_obs) )
        
        new_A,new_B,new_pi = one_iter(lst_of_obs, A, B, pi)

        new_score = sum( (np.log(forward_prob_table(obs, new_A, new_B, new_pi)[1]) for obs in lst_of_obs) )
        
        print "iteration %d, score %f" %(iteration, new_score)
        
        if convergent(old_score, new_score):
            return new_A, new_B, new_pi

        A, B, pi = new_A, new_B, new_pi

        #to prevent memory usage explode
        clear_memoization()

        iteration += 1        