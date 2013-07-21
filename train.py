from __future__ import division
from collections import Counter

from lmatrix import LMatrix

from util import hashdict

def init_estimate(Q, obs_list):
    """
    (state set, observation list) -> A, B, pi

    given a list of observation, initial estimate of the parameter
    """
    #get the state set and the output vocabulary
    V = set( (a[0] for obs in obs_list for a in obs) )
    
    #calculating the state transition matrix
    #get the bigrams, (from-state,to-state) list
    transition_tuple_list = [(prev_ob.tag, cur_ob.tag) for obs in obs_list for prev_ob, cur_ob in zip(obs[:-1], obs[1:])]

    #calculate the frequency
    transition_freq = Counter(transition_tuple_list)

    A_unnormalized = LMatrix(Q)

    for fs in Q:
        for ts in Q:
            A_unnormalized[fs, ts] = transition_freq[(fs, ts)]

    #normalize it
    rc,cc =  A_unnormalized.shape
    A = A_unnormalized / A_unnormalized.sum(1).reshape(rc,1).repeat(cc,1)

    
    #calculating the emission prob matrix
    #emission tuples, (state, observation ) list
    emission_tuple_list = [(ob.tag, ob.word) for obs in obs_list for ob in obs]
    
    #the same above
    emission_freq = Counter(emission_tuple_list)
    
    B_unnormalized = LMatrix(Q, V)
    
    for s in Q:
        for o in V:
            B_unnormalized[s,o] = emission_freq[(s,o)]

    #normalize it
    rc,cc =  B_unnormalized.shape
    B = B_unnormalized / B_unnormalized.sum(1).reshape(rc,1).repeat(cc,1)

    #calculating pi
    #starting state list
    start_states = [obs[0][1] for obs in obs_list]
    
    #the same above
    start_state_freq = Counter(start_states)

    pi = dict()
    for s in Q:
        pi[s] = start_state_freq[s] / sum(start_state_freq.values())
        
    pi = hashdict(pi)
    
    return A, B, pi


if __name__ == '__main__':
    from util import read_annotation
    
    obs_list = read_annotation(open("data/annotated.json", "r"))
    A, B, pi = init_estimate(("O", "B", "C"), obs_list)

    from baum_welch import baum_welch
    
    A, B, pi = baum_welch(obs_list, A, B, pi)
