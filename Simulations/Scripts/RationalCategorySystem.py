
'''
'''
import string
import numpy as np
from scipy.special import logsumexp
import scipy.stats.distributions as dist
from collections import defaultdict
import numpy.random as nprand
import random
import os


VERBOSE = False 

def get_tdist(df, mu, sigma, x):
    tdist = dist.t([df])
    return tdist.pdf((x-mu)/sigma)

def stims_equal(stim1, stim2):
    return all([s[0]==s[1] for s in zip(stim1, stim2)])
class RationalModel:
    '''
    Implementation for continuous features only
    updated so can start with an already populated system 
    Adapted from: https://github.com/johnmcdonnell/Rational-Models-of-Categorization/blob/master/particle.py
    '''

    def __init__(self, c, mu_0, sigmasq_0, lambda_0, a_0, partition=None, stimuli=None, max_new_clusters=np.inf):
        if partition is None or stimuli is None:
            self.partition = []    
            self.stimuli = []
            self.N = 0
            self.clusters = 0
            self.label_to_cluster = defaultdict(lambda: None)
            self.cluster_to_label = defaultdict(lambda: None)
        else:
            self.initialize_from_existing(stimuli, partition)
        self.c = c # coupling parameter == probability items are from the same category 
        self.mu_0 = mu_0 # prior on feature means
        self.sigmasq_0 = sigmasq_0 # prior on feature variances 
        self.lambda_0 = lambda_0 # strength of mean prior
        self.a_0 = a_0 # strength of variance prior
        self.max_clusters = self.clusters + max_new_clusters

    def initialize_from_existing(self, stimuli, partition):
        if len(stimuli) != len(partition):
            raise ValueError("Stimuli and partition must be same length")
        self.stimuli = stimuli
        self.N = len(stimuli)
        unique_labels = sorted(set(partition), key=str)
        self.label_to_cluster = defaultdict(lambda:None, {label: i for i, label in enumerate(unique_labels)})
        self.cluster_to_label = defaultdict(lambda:None, {i: label for i, label in enumerate(unique_labels)})
        self.partition = [self.label_to_cluster[label] for label in partition]
        self.clusters = len(unique_labels)


    '''
    Makes new category labels: A, B, ..., Z, AA, AB, ...
    '''
    def make_new_label(self, n):
        letters = string.ascii_uppercase
        label = ""
        while True:
            n, r = divmod(n, 26)
            label = letters[r] + label
            if n == 0:
                break
            n -= 1
        return label


    def find_distribution(self, k, i, lambda_i, a_i, n):
        """
        For a given cluster computes the current mean and variance.
        """
        items = []
        for index in range(len(self.partition)):
            if self.partition[index] == k:
                items.append( self.stimuli[index][i] )
        
        if n>0:
            xbar = np.mean(items)
        else:
            xbar = 0
        if n > 1:
            var = np.var(items)
        else:
            var = 0
        
        mui = ((self.lambda_0[i]*self.mu_0[i]) + (n * xbar)) / lambda_i # should be divided by lambda_i not a_i
        if n == 0:
            sigmasq_i=self.sigmasq_0[i]
        else:
            sigmasq_i = ((self.a_0[i]*self.sigmasq_0[i]) +
                         ((n-1.0) * var) + ((self.lambda_0[i]*n)/lambda_i) * ((self.mu_0[i] - xbar)**2)) / a_i
        return mui, sigmasq_i

    def prob_density(self, k, i, x):
        if k == self.clusters:
            n = 0
        else:
            n = sum([item == k for item in self.partition])
        
        lambda_i = self.lambda_0[i] + n 
        a_i = self.a_0[i] + n
        mu_i, sigmasq_i = self.find_distribution(k, i, lambda_i, a_i, n)
        prob = get_tdist(a_i, mu_i, np.sqrt(sigmasq_i * (1.0 + (1.0/lambda_i))), x)
        return prob

    def stimulus_prob(self, stim_idx, k):
        stimulus = self.stimuli[stim_idx]
        prob_jks = []
        for i in range(len(stimulus)):
            prob_jks.append(self.prob_density(k, i, stimulus[i]))
        return(np.prod(prob_jks))
    
    def compute_posterior(self, stimulus):
        if self.clusters < self.max_clusters:
            pk  = np.empty(self.clusters+1)
            pfk = np.empty(self.clusters+1)
        else:
            pk  = np.empty(self.clusters)
            pfk = np.empty(self.clusters)

        for k in range(self.clusters):
            pk[k] = np.log((self.c * sum([item == k for item in self.partition])) / ((1.0-self.c) + (self.c*self.N)))
            pfk[k] = self.stimulus_prob(stimulus, k)
        
        if self.clusters < self.max_clusters:
            pk[self.clusters] = np.log((1.0-self.c) / (( 1.0-self.c ) + (self.c *self.N)))
            pfk[self.clusters] = self.stimulus_prob(stimulus, self.clusters)
        num = pk + pfk
        denom = logsumexp(num)
        pkf = num - denom
        if VERBOSE:
            print("p(k)s: ", np.exp(pk), 'Sum: ', np.sum(np.exp(pk)))
            print("p(f|k)s: ", np.exp(pfk))
            print("p(k|f): ", np.exp(pkf), 'Sum: ', np.sum(np.exp(pkf)))
        self.current_posterior = pkf
        self.last_stimuli = self.stimuli[stimulus]
        return pkf

    def register_item(self, stim):
        self.stimuli.append(stim)
        self.partition.append(-1)
        return len(self.stimuli) - 1
    
    def get_item_likelihood(self, stim, label):
        stim_idx = self.register_item(stim)
        posterior = self.compute_posterior(stim_idx)
        if self.label_to_cluster[label] is None:
            choice_idx = self.clusters
            self.clusters += 1
            self.label_to_cluster[label] = choice_idx
            if self.clusters == self.max_clusters:
                self.c = 1.0
        else:
            choice_idx = self.label_to_cluster[label]
        prob = posterior[choice_idx]
        self.partition[stim_idx] = choice_idx
        self.N += 1
        return prob
