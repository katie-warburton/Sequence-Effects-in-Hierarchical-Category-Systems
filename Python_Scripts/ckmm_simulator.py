import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import OrderedCategorySystem as OCS

F =  [12, 13, 15, 14, 16, 18, 17, 19, 10]
B =  [20, 19, 17, 18, 16, 14, 15, 13, 12]
M1 =  [16, 17, 15, 18, 14, 19, 13, 10, 12]
M2 = [16, 15, 17, 14, 18, 13, 19, 12, 10]

NEW = [i for i in range(9, 14)]
ALL = NEW + [1, 3, 19, 31]

SHIFT = 3
DISTRACTORS = [1, 3, 19, 31]

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

def get_item_set(order, loc):
    if order in ['F', 'B', 'M']: # Sequence
        if order == 'F':
            items = F
        elif order == 'B':
            items = B
        else: 
            items = random.choice([M1, M2])
        random.shuffle(DISTRACTORS)
        if loc == 'L':
            items = [i-SHIFT for i in items]
        elif loc == 'R':
            items = [i+SHIFT for i in items]
        return items[0:1] + DISTRACTORS[0:1] + items[1:4] + DISTRACTORS[1:2] + items[4:5] + DISTRACTORS[2:3] + items[5:8] + DISTRACTORS[3:4] + items[8:9]
    else: # All at once 
        items = F
        if loc == 'L':
            items = [i-SHIFT for i in items]
        elif loc == 'R':
            items = [i+SHIFT for i in items]
        final_set = items + DISTRACTORS
        random.shuffle(final_set)
        return final_set
    
def simulate_trials(p_data, conditions, temps, alphas):
    trials = []
    D, item_hash = OCS.get_distance_mat(item_space)
    treeCache = defaultdict(lambda: None)
    for i in range(len(conditions)):
        code, p_id = conditions[i].split('_')
        d, l, o = list(code)
        items = get_item_set(o, l)
        p_id =  f's{int(p_id):03d}'
        if d == '2':
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree2D.json')
            score, cat_choices = OCS.greedy_categorizer_softmax(syst, items, D, treeCache, temps[0], alphas[0])

        else:
            syst = OCS.CategorySystem(item_hash, 'Python_Scripts/BaseSystems/tree3D.json')
            score, cat_choices = OCS.greedy_categorizer_softmax(syst, items, D, treeCache, temps[1], alphas[1])

        cat_choices['SCORE'] = score
        cat_choices['DEPTH'] = int(d)
        cat_choices['LOC'] = l
        cat_choices['ORDER'] = o.lower()
        trials.append(cat_choices)




def get_trial_codes(fp):
    allParticipants = pd.read_csv(fp)
    allParticipants =  allParticipants[(allParticipants['ATTEMPTS'] < 3) & (allParticipants['TOTAL_ERRORS'] < 4)]
    participants = allParticipants['P_ID'].tolist()

    trials_df = pd.read_csv('Data/trial_data.csv')    
    trials_df = trials_df[trials_df['P_ID'].isin(participants)][['DEPTH', 'LOC', 'ORDER']]

    codes = trials_df["DEPTH"].astype(str) + trials_df["LOC"] + trials_df['ORDER'].str.upper() + '_' + trials_df.index.astype(str)
    return codes.tolist()


def simulate(sims, codes, temps, alphas):
    master_seed = 13
    rng = np.random.default_rng(master_seed)
    seeds = rng.integers(0, 2**31, size=sims).tolist()
    p_ids = {f's{int(c.split('_')[1]):03d}': {'HAS_X': [], 'SAME_AS_FIRST': []} for c in codes}

    for i, seed in enumerate(seeds):
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        random.shuffle(codes)
        simulate_trials(codes, temps, alphas)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Simulate categorization with CKMM")
#     parser.add_argument('n', type=int, help='The number of simulations to run.')
#     parser.add_argument('--temps', type=tuple, default=(4.0, 1.0), help='Temperature for 2-level and 3-level conditions. Default: (4.0, 1.0).')
#     parser.add_argument('--alpha', type=tuple, default=(0.05, 0.14), help='Second and third level mixture parameter. Default: (0.05, 0.14).')
#     args = parser.parse_args()
#     codes = get_trial_codes('../Data/trial_data.csv')
    # data = simulate(args.n, codes, args.temps, args.alphas)

codes = get_trial_codes('Data/participant_data.csv')
simulate(1, codes, (1.0, 4.0), (0.05, 0.14))