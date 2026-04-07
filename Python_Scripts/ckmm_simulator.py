import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import OrderedCategorySystem as OCS

ITEMS = ['I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23']

LEFT = ITEMS[:9]
CENTRE = ITEMS[3:12]
RIGHT = ITEMS[6:]

LOCS = [('L', LEFT),
        ('C', CENTRE), 
        ('R', RIGHT)]
ORDERS = [('f', 0, [1, 1, 3, 4, 5, 6, 7, 8]),
          ('m', 4, [0, 1, 1, 3, 5, 6, 7, 8]),
          ('b', 8, [0, 1, 1, 3, 4, 5, 6, 7])]

item_space = [i for i in range(1, 32)]

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
        idx = rng.permutation(len(final_set))
        final_set = [final_set[i] for i in idx]
        return final_set

def simulate_trials(conditions, softmax, temps=(1, 1), alphas=(0, 0), seed=18):
    item_space = [i for i in range(1, 32)]
    D, item_hash = OCS.get_distance_mat(item_space)
    treeCache = defaultdict(lambda: None)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(conditions))
    trials = []
    for i in range(len(conditions)):
        rng = np.random.default_rng(child_seeds[i])
        code = conditions[i]
        d, l, o = list(code)
        items = get_item_set(o, l, rng=rng)
        if d == '2':
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree2D.json')
        else:
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree3D.json')
        if softmax:
            if d == '2':
                score, cat_choices = OCS.greedy_categorizer_softmax(syst, items, D, treeCache, temps[0], alphas[0], rng=rng)
            else:
                score, cat_choices = OCS.greedy_categorizer_softmax(syst, items, D, treeCache, temps[1], alphas[1], rng=rng)
        else:
            score, cat_choices = OCS.greedy_categorizer(syst, items, D, treeCache)
        cat_choices['SCORE'] = score
        cat_choices['DEPTH'] = int(d)
        cat_choices['LOC'] = l
        cat_choices['ORDER'] = o.lower()
        cat_choices['P_ID'] = f's{i:03d}'
        cat_choices['POOL'] = 'simulated'
        cat_choices['STIMULI'] = 'simulated'
        trials.append(cat_choices)
        if i % 1000 == 0:
            print(f'{i+1} iteration(s) completed')
    df = pd.DataFrame.from_dict(trials)
    df = df[['P_ID', 'DEPTH', 'LOC', 'ORDER'] + [i for i in range(9, 24)] + [1, 3, 29, 31] + ['STIMULI', 'POOL', 'SCORE']]
    df.rename(columns={i:f'I{i:02}' for i in range(9, 24)}, inplace=True)
    df.rename(columns={1: 'I01', 3: 'I03', 29: 'I29', 31: 'I31'}, inplace=True)
    return df

def get_trial_codes(fp):
    allParticipants = pd.read_csv(fp)
    allParticipants =  allParticipants[(allParticipants['ATTEMPTS'] < 3) & (allParticipants['TOTAL_ERRORS'] < 4)]
    participants = allParticipants['P_ID'].tolist()

    trials_df = pd.read_csv('Data/trial_data.csv')    
    trials_df = trials_df[trials_df['P_ID'].isin(participants)][['DEPTH', 'LOC', 'ORDER']]

    codes = trials_df["DEPTH"].astype(str) + trials_df["LOC"] + trials_df['ORDER'].str.upper() + '_' + trials_df.index.astype(str)
    return codes.tolist()

def simulate(sims, codes, temps, alphas):
    master_seed = 99
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=sims).tolist()
    p_ids = {f's{int(c.split('_')[1]):03d}': {'HAS_X': [], 'SAME_AS_FIRST': []} for c in codes}
    for i, seed in enumerate(seeds):
        print(seed)        
        random.shuffle(codes)
        simulate_trials(codes, temps, alphas, seed=seed)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Simulate categorization with CKMM")
#     parser.add_argument('n', type=int, help='The number of simulations to run.')
#     parser.add_argument('--temps', type=tuple, default=(4.0, 1.0), help='Temperature for 2-level and 3-level conditions. Default: (4.0, 1.0).')
#     parser.add_argument('--alpha', type=tuple, default=(0.05, 0.14), help='Second and third level mixture parameter. Default: (0.05, 0.14).')
#     args = parser.parse_args()
#     codes = get_trial_codes('../Data/trial_data.csv')
    # data = simulate(args.n, codes, args.temps, args.alphas)

codes = get_trial_codes('Data/participant_data.csv')
simulate(1, codes, (4.0, 1.0), (0.05, 0.14))