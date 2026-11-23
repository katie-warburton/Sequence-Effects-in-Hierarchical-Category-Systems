import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import OrderedCategorySystem as OCS

NOVEL_ITEMS = ['I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23']

'''
Generate the sequence of items used in a single simulated trial based on its location (loc) and order condition. 
return the set as a list of item numbers. 
'''
def get_item_set(order, loc, rng=None):
    shift = 3
    distractors = [1, 3, 29, 31]
    if rng is None:
        rng = np.random.default_rng()
    if order in ['F', 'B', 'M']: # Sequence
        if order == 'F':
            items =  [12, 13, 15, 14, 16, 18, 17, 19, 20]
        elif order == 'B':
            items = [20, 19, 17, 18, 16, 14, 15, 13, 12]
        else:
            # Like participants, the model randomly sees one of the two possible middle orders. 
            M1 =  [16, 17, 15, 18, 14, 19, 13, 20, 12]
            M2 = [16, 15, 17, 14, 18, 13, 19, 12, 20]
            items = [M1, M2][rng.integers(2)]
        idx = rng.permutation(len(distractors))
        distractors = [distractors[i] for i in idx]
        if loc == 'L':
            items = [i-shift for i in items]
        elif loc == 'R':
            items = [i+shift for i in items]
        return items[0:1] + distractors[0:1] + items[1:4] + distractors[1:2] + items[4:5] + distractors[2:3] + items[5:8] + distractors[3:4] + items[8:9]
    else: # All at once 
        items = [12, 13, 15, 14, 16, 18, 17, 19, 20]
        if loc == 'L':
            items = [i-shift for i in items]
        elif loc == 'R':
            items = [i+shift for i in items]
        final_set = items + distractors
        # we simulate the all-at-once condition by randomizing stimulus order. 
        idx = rng.permutation(len(final_set))
        final_set = [final_set[i] for i in idx]
        return final_set

'''
Simulate the experiment using ordered CKMM. 
Softmax controls whether the deterministic or softmax variant of ordered CKMM are run,
temp and alpha reflect model parameters, and seed sets the random state for replicability. 
A pandas dataframe equivalentally formatted to the participant data is returned. 
'''
def simulate_trials(conditions, softmax, temp=1.0, alpha=0, seed=1999):
    item_space = [i for i in range(1, 32)]
    D, item_hash = OCS.get_distance_mat(item_space)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(conditions))
    trials = []
    for i in range(len(conditions)):
        treeCache = defaultdict(lambda: None)
        rng = np.random.default_rng(child_seeds[i])
        code = conditions[i]
        d, l, o = list(code) # depth, loc, order 
        items = get_item_set(o, l, rng=rng)
        if d == '2':
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree2D.json')
        else:
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree3D.json')
        if softmax:
            score, cat_choices = OCS.greedy_categorizer_softmax(syst, items, D, treeCache, temp, alpha, rng=rng)
        else: # if deterministic
            score, cat_choices = OCS.greedy_categorizer(syst, items, D, treeCache)
        # store trial level data in an equivalent format to the participant data
        cat_choices['SCORE'] = score
        cat_choices['DEPTH'] = int(d)
        cat_choices['LOC'] = l
        cat_choices['ORDER'] = o.lower()
        cat_choices['P_ID'] = f's{i:03d}'
        cat_choices['POOL'] = 'simulated'
        cat_choices['STIMULI'] = 'simulated'
        trials.append(cat_choices)
        if i % 200 == 0:
            print(f'{i+1} iteration(s) completed')
    df = pd.DataFrame.from_dict(trials)
    df = df[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [i for i in range(9, 24)] + [1, 3, 29, 31] + ['STIMULI', 'POOL', 'SCORE']]
    df.rename(columns={i:f'I{i:02}' for i in range(9, 24)}, inplace=True)
    df.rename(columns={1: 'I01', 3: 'I03', 29: 'I29', 31: 'I31'}, inplace=True)
    return df

def run_simulation(softmax, temp, alpha, seed=1999):
    # get participant_list
    allParticipants = pd.read_csv('Data/participant_data.csv')
    allParticipants =  allParticipants[(allParticipants['ATTEMPTS'] < 3) & (allParticipants['TOTAL_ERRORS'] < 4)]
    participants = allParticipants['P_ID'].tolist()

    # get participant trial codes so the number of each trial type is equivalent between simulated and participant data
    trials_df = pd.read_csv('Data/trial_data.csv')    
    trials_df = trials_df[trials_df['P_ID'].isin(participants)][['DEPTH', 'LOC', 'ORDER']]
    codes = trials_df["DEPTH"].astype(str) + trials_df["LOC"] + trials_df['ORDER'].str.upper()
    codes = codes.tolist()

    # simulate the full experiment 
    df_sm_full = simulate_trials(codes, softmax, temp=temp, alpha=alpha, seed=seed)
    df_sm_full.to_csv('Data/sim_trial_data.csv', index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate categorization with CKMM. Default parameters will replicate the data used in the paper.")
    parser.add_argument('softmax', type=bool, help='Whether to run ordered CKMM as a greedy deterministic model or stochastically with a softmax layer.')
    parser.add_argument('--temp', type=float, default= 1.0, help='Temperature parameter. Default: 1.0.')
    parser.add_argument('--alpha', type=float, default=0.15, help='Mixture parameter. Default: 0.15.')
    parser.add_argument('--seed', type=int, default=1999, help="Random seed. Default: 1999")
    args = parser.parse_args()
    run_simulation(args.softmax, args.temp, args.alpha, args.seed)