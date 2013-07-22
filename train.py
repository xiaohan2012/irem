from __future__ import division
from collections import Counter

import numpy as np

from lmatrix import LMatrix

from util import hashdict, get_V

def init_estimate(Q, annotation_list, discount = False, V = None, vfreq = None):
    """
    (state set, observation list, output vocab, vocab freq) -> A, B, pi

    given a list of observation, initial estimate of the parameter,
    if V and vfreq are given, use the discounted version
    """
    
    #get V
    if V is None:
        V = get_V(annotation_list)
    
    #calculating the state transition matrix
    #get the bigrams, (from-state,to-state) list
    transition_tuple_list = [(prev_ob.tag, cur_ob.tag) for obs in annotation_list for prev_ob, cur_ob in zip(obs[:-1], obs[1:])]

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
    emission_tuple_list = [(ob.tag, ob.word) for obs in annotation_list for ob in obs]
    
    #the same above
    emission_freq = Counter(emission_tuple_list)
    
    B_unnormalized = LMatrix(Q, V)
    
    for s in Q:
        for o in V:
            B_unnormalized[s,o] = emission_freq[(s,o)]

    if not discount:
        #not using discounting
        #normalize it
        rc,cc =  B_unnormalized.shape
        B = B_unnormalized / B_unnormalized.sum(1).reshape(rc,1).repeat(cc,1)
    else: #using discounting
        B = discounted(B_unnormalized, V, vfreq)

    #calculating pi
    #starting state list
    start_states = [obs[0][1] for obs in annotation_list]
    
    #the same above
    start_state_freq = Counter(start_states)

    pi = dict()
    for s in Q:
        pi[s] = start_state_freq[s] / sum(start_state_freq.values())
        
    pi = hashdict(pi)
    
    return A, B, pi

def word_freq(annotation_list, obs_list):
    """given the annotation list and the obs list, return the word frequency table"""
    return Counter( [pair.word for sent in annotation_list for pair in sent] + [ob for obs in obs_list for ob in obs] )
    
def discounted(B, V, V_freq):
    """
    get the discounted version of the `B` emission prob matrix
    
    B: the original **freq**(not prob) matrix
    V: the vocabulary that needs to be included in B
    V_freq: the word frequency table(useful for the discounting)
    """
    non_zero_idx = (np.array(B) != 0)

    #make count*
    B_star = B.copy()
    
    B_star[non_zero_idx] = B_star[non_zero_idx] - 0.5

    #init the discounted B matrix
    dB = LMatrix(B.rlabels, V)
    
    for s in dB.rlabels:
        
        #set of words, such that word `o`'s count(`s`,`o`) > 0 in B
        A_set = set( filter(lambda w: B[s,w] > 0, B.clabels) )
        
        #set of words, such that word `o`'s count(`s`,`o`) = 0 in B
        #simple, set substraction operation 
        B_set = V - A_set
        
        missing_prob_mass = 1 - B_star[s,:].sum() / B[s,:].sum()
        
        for o in dB.clabels:

            if o in A_set:
                dB[s,o] = B_star[s,o] / B[s,:].sum()
            else:
                dB[s,o] = V_freq[o] / sum( (V_freq[w] for w in B_set) ) * missing_prob_mass

    return dB
    
if __name__ == '__main__':
    from util import read_annotation, sample_observations_from_file

    #for initial estimate, get the annotation list
    annotation_list = read_annotation(open("data/annotated.json", "r"))

    #get the observation list
    obs_list = sample_observations_from_file(open("data/ingredients5000.json", "r"), 2500)

    freq = word_freq(annotation_list, obs_list)
    
    #get Q and V
    Q = ("O", "B", "C")
    V = get_V(annotation_list, obs_list)

    print "init estimating..."
    A, B, pi = init_estimate(Q, annotation_list, discount = True, V = V, vfreq = freq)
    
    from baum_welch import baum_welch

    print "learning..."
    A, B, pi = baum_welch(obs_list, A, B, pi)

    