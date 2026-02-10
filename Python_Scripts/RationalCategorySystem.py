import numpy as np
import copy
from scipy.special import logsumexp
from scipy.stats import t
from collections import defaultdict

VERBOSE = False 

def get_tdist(x, a, mu, sigmasq, lambd):
    scale = np.sqrt(sigmasq * (1.0 + (1.0/lambd)))
    return t.logpdf(x, df=a, loc=mu, scale=scale)
class RationalModel:
    '''
    Implementation for continuous features only
    updated so can start with an already populated system 
    Adapted from: https://github.com/johnmcdonnell/Rational-Models-of-Categorization/blob/master/particle.py
    -- modified such that once participants make the maximum number of categories the coupling constraint goes to 1.0
    and no probability of creating a new category is calculated
    -- can be greedy or probabilistic 
    -- can make a mixture model with uniform category distribution
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
        self.stimuli = copy.deepcopy(stimuli)
        self.N = len(stimuli)
        unique_labels = sorted(set(partition), key=str)
        self.label_to_cluster = defaultdict(lambda:None, {label: i for i, label in enumerate(unique_labels)})
        self.cluster_to_label = defaultdict(lambda:None, {i: label for i, label in enumerate(unique_labels)})
        self.partition = [self.label_to_cluster[label] for label in partition]
        self.clusters = len(unique_labels)

    def find_distribution(self, k, i, lambda_i, a_i, n):
        """
        For a given cluster computes the current mean and variance.
        """
        items = [self.stimuli[idx][i] 
                 for idx, cl in enumerate(self.partition)
                 if cl == k]
        if n>0:
            xbar = np.mean(items)
        else:
            xbar = 0
        if n > 1:
            # POPULATION OR SAMPLE (ddof = 1 for sample mean)
            var = np.var(items, ddof=1)
        else:
            var = 0
        
        mu_i = ((self.lambda_0[i]*self.mu_0[i]) + (n * xbar)) / lambda_i
        if n == 0:
            sigmasq_i=self.sigmasq_0[i]
        else:
            sigmasq_i = ((self.a_0[i]*self.sigmasq_0[i]) + ((n-1.0) * var) +
                         ((self.lambda_0[i]*n)/lambda_i) * ((self.mu_0[i] - xbar)**2)) / a_i
        return mu_i, sigmasq_i

    def prob_density(self, k, i, x):
        if k == self.clusters:
            n = 0
        else:
            n = sum([item == k for item in self.partition])
        lambda_i = self.lambda_0[i] + n 
        a_i = self.a_0[i] + n
        mu_i, sigmasq_i = self.find_distribution(k, i, lambda_i, a_i, n)
        prob = get_tdist(x, a_i, mu_i, sigmasq_i, lambda_i)
        return prob

    def stimulus_prob(self, stim_idx, k):
        stim = self.stimuli[stim_idx]
        return sum(self.prob_density(k, i, stim[i]) for i in range(len(stim)))
    
    def compute_posterior(self, stim_idx):
        if self.clusters < self.max_clusters:
            pk  = np.empty(self.clusters+1)
            pfk = np.empty(self.clusters+1)
        else:
            pk  = np.empty(self.clusters)
            pfk = np.empty(self.clusters)

        for k in range(self.clusters):
            pk[k] = np.log(
                (self.c * sum([cl == k for cl in self.partition])) /
                ((1.0 - self.c) + (self.c * self.N))
            )
            pfk[k] = self.stimulus_prob(stim_idx, k)
        if self.clusters < self.max_clusters:
            pk[self.clusters] = np.log(
                (1.0 - self.c) / ((1.0 - self.c) + (self.c * self.N))
            )
            pfk[self.clusters] = self.stimulus_prob(stim_idx, self.clusters)
        num = pk + pfk
        denom = logsumexp(num)
        pkf = num - denom
        if VERBOSE:
            print("p(k): ", np.exp(pk), " sum=", np.sum(np.exp(pk)))
            print("p(f|k): ", np.exp(pfk))
            print("p(k|f): ", np.exp(pkf), " sum=", np.sum(np.exp(pkf)))
        self.current_posterior = pkf
        return pkf
    
    def get_item_likelihood(self, stim, label):
        # add stimulus to system
        self.stimuli.append(stim) 
        stim_idx = len(self.stimuli) - 1  
        # compute prob distribution
        posterior = self.compute_posterior(stim_idx)
        # get choice idx and update system
        # double check this part'
        if self.label_to_cluster[label] is None:
            choice_idx = self.clusters
            self.label_to_cluster[label] = choice_idx
            self.cluster_to_label[choice_idx] = label
            self.clusters += 1
            if self.clusters == self.max_clusters:
                self.c = 1.0  # force c to 1 if at max clusters
        else:
            choice_idx = self.label_to_cluster[label]
        self.partition.append(choice_idx)
        self.N += 1
        return posterior, choice_idx
    

def get_trial_loglike(model, choices, order, item_space, alpha=0.0, uniform=None):
    log_like = 0
    log_probs = []
    for t in range(len(order)):
        it = order[f't{t+1:02}']
        choice = choices[it]
        item_rep = item_space[it-1]
        if alpha == 0 or uniform is None: # can use this to precompute probs -- and then only need to vary alphas :)
            prob_dist, choice_idx = model.get_item_likelihood(item_rep, choice)
            log_like += prob_dist[choice_idx]
            log_probs.append(prob_dist[choice_idx])
        else:
            prob_dist, choice_idx = model.get_item_likelihood(item_rep, choice)
            mixture = np.log(((1-alpha)*np.exp(prob_dist[choice_idx])) + alpha*uniform)
            log_like += mixture
            log_probs.append(mixture)
    return log_like, log_probs

def get_participant_loglike(data, item_space, depth, c=0.5, alpha=0.0):
    total_ll = 0.0
    log_probs = []
    existing_items = [item_space[i] for i in [1, 2, 3, 5, 6, 7, 23, 24, 25, 27, 28, 29]]
    mu_0 = [np.mean(item_space)]
    var_0 = [np.var(item_space)]
    i = 0
    for trial in data:
        order = trial['SEQUENCE']
        choices = trial['CHOICES']
        if depth == 3:
            syst = ['L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'R2', 'R2', 'R2', 'R1', 'R1', 'R1']
            max_new = 3
            uniform = 1/7
        else:
            syst = ['L',  'L',  'L',  'L',  'L',  'L',  'R',  'R',  'R',  'R',  'R',  'R']
            max_new = 1
            uniform = 1/3
        rcm = RationalModel(c, mu_0, var_0,  np.array([1.0]),  np.array([1.0]), partition=syst, stimuli=existing_items, max_new_clusters=max_new)
        trial_ll, trial_log_probs = get_trial_loglike(rcm, choices, order, item_space, alpha, uniform)
        total_ll += trial_ll
        log_probs.extend(trial_log_probs)
        i += 1
    return total_ll, log_probs

def get_total_log_like(trial_data, item_space, depth, c=0.5, alpha=0.0):
    total_ll = 0.0
    log_probs = []
    for _, p_data in trial_data.items():
        participant_ll, participant_log_probs =  get_participant_loglike(p_data, item_space, depth, c, alpha)
        total_ll += participant_ll
        log_probs.extend(participant_log_probs)
    return total_ll, np.array(log_probs)

def get_loglike_and_n(trial_data, item_space, depth, c=0.5, alpha=0.0):
    total_ll = 0.0
    n_trials = 0
    for _, p_data in trial_data.items():
        participant_ll, _ =  get_participant_loglike(p_data, item_space, depth, c, alpha)
        total_ll += participant_ll
        n_trials += len(p_data)
    return total_ll, n_trials

def format_rcm_data(trial_data):
    rcmData = {t['P_ID']:[] for t in trial_data}
    for t in trial_data:
        item_assignments = {int(key[1:]): val for key, val in t['ITEMS'].items()}
        sequence = {key: int(val) for key, val in t['SEQUENCE'].items()}
        rcmData[t['P_ID']].append({'CHOICES': item_assignments, 'SEQUENCE': sequence, 'LOC': t['LOC'], 'ORDER': t['ORDER'], 'STIMULI': t['STIMULI']})
    return rcmData

def find_best_params(data, item_space, depth, params, determ=False):
    best_ll = -np.inf
    c_vals, alphas = params
    best_c, best_a = None, None
    for c in c_vals:
        prev_ll = None
        base_ll, log_probs = get_total_log_like(data, item_space, depth, c, 0)
        for a in alphas:
            if a == 0:
                if determ:
                    raise ValueError('Cannot be deterministic if alpha=0')
                else:
                    ll = base_ll
            else:
                if determ: ## FIX!!!!!!!
                    # probs = (log_probs == log_probs.max(axis=1)[:,None]).astype(int)
                    # print(probs)
                    # probs = probs/probs.sum(axis=1)
                    # print(probs)
                    probs = np.zeros_like(log_probs)
                    probs[log_probs.argmax()] = 1
                    print(probs)
                    probs = probs/probs.sum()
                    print(probs)
                else:
                    probs = np.exp(log_probs)
                if depth == 2:
                    ll = np.sum(np.log((1-a)*probs + (a/3)))
                else:
                    ll = np.sum(np.log((1-a)*probs + (a/7)))
            if ll > best_ll:
                best_ll = ll
                best_c, best_a = c, a
            # Alphas are concave - can stop if it starts decreasing
            if prev_ll is not None and ll < prev_ll:
                break
            prev_ll = ll
        stats = (best_c, best_a, best_ll)
    return stats