import copy
import pandas as pd
import numpy as np
from scipy.spatial import distance

def first_char(val):
    if val is not np.nan:
        return val[0]
    else:
        return np.nan

def get_level2_cat_stats(df_orig, items, include_disctractors=False):
    if include_disctractors:
        div = 13
    else:
        div = 9
    df = copy.deepcopy(df_orig)
    df[items] = df[items].map(first_char)
    df['NUM_L'] = df[df[items] == 'L'].count(axis=1)
    df['NUM_R'] = df[df[items] == 'R'].count(axis=1)
    df['NUM_X'] = df[df[items] == 'X'].count(axis=1)
    df['PROP_L'] = df['NUM_L']/div
    df['PROP_R'] = df['NUM_R']/div
    df['PROP_X'] = df['NUM_X']/div
    df['HAS_X'] = df['NUM_X'].apply(lambda x: 1 if x > 0 else 0)
    return df

def get_seq_df(df, items, loc, orders, include_distractors=False):
    if include_distractors:
        div = 12
    else:
        div = 8
    loc_df = df[df['LOC'] == loc]
    items = np.array(items)
    seq_dfs = []
    # long_dfs = []
    for lab, first_idx, other_idx in orders:
       first = items[first_idx]
       if include_distractors: #ok to add them to the end because they are never first
            other = items[other_idx].tolist() + ['I01', 'I03', 'I29', 'I31']
       else:
           other = items[other_idx].tolist()
       item_subset = loc_df[loc_df['ORDER'] == lab]
       same_as_first = item_subset[[first] + other].apply(lambda x: x[other] == x[first], axis=1)*1
       same_as_first['PROP_SAME'] = same_as_first[other].sum(axis=1) / div
       same_as_first['FIRST_IT'] = first
       subseq_df = pd.concat([item_subset[['P_ID', 'STIMULI', 'POOL', 'DEPTH', 'LOC', 'ORDER']], item_subset[first], same_as_first], axis=1)
       subseq_df.rename(columns={first: 'FIRST'}, inplace=True)
    #    long_seq = subseq_df.melt(id_vars = ['P_ID', 'STIMULI', 'POOL', 'DEPTH', 'LOC', 'ORDER', 'FIRST', 'FIRST_IT'], value_vars=other, var_name='ITEM', value_name='SAME')
    #    long_seq.rename(columns={first: 'FIRST_CAT'}, inplace=True)
       subseq_df.rename(columns={other[i] : f'OTH_{i+1}' for i in range(len(other))}, inplace=True)
       seq_dfs.append(subseq_df)
    #    long_dfs.append(long_seq)
    return pd.concat(seq_dfs)#, pd.concat(long_dfs)

def get_seq_data(df, locs, orders, include_distractors=False):
    dfs_by_loc = []
    # long_by_loc = []
    for label, items in locs:
        seq_df = get_seq_df(df, items, label, orders, include_distractors)
        dfs_by_loc.append(seq_df)
        # long_by_loc.append(long_df)
    seq_df = pd.concat(dfs_by_loc)
    # long_df = pd.concat(long_by_loc)
    return seq_df.reset_index().drop('index', axis=1)#, long_df.reset_index().drop('index', axis=1)

def jsd_no_nan(dist1, dist2):
    with np.errstate(invalid='ignore'):
        dist = distance.jensenshannon(dist1, dist2)
        if np.isnan(dist):
            dist = 0.0
    return dist

def get_jsds(df, loc):
    df_at_loc = df[df['LOC'] == loc]
    dist_a = np.array(df_at_loc[df_at_loc['ORDER'] == 'a'][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)
    dist_f = np.array(df_at_loc[df_at_loc['ORDER'] == 'f'][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)
    dist_m = np.array(df_at_loc[df_at_loc['ORDER'] == 'm'][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)
    dist_b = np.array(df_at_loc[df_at_loc['ORDER'] == 'b'][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)

    jsd_f = jsd_no_nan(dist_f, dist_a)
    jsd_m = jsd_no_nan(dist_m, dist_a)
    jsd_b = jsd_no_nan(dist_b, dist_a)
    return jsd_f, jsd_m, jsd_b

def perm_test(df, loc, it, seed=13):
    jsd_f, jsd_m, jsd_b = get_jsds(df, loc)
    if loc == 'L':
        diff1 = jsd_b - jsd_f
        diff2 = jsd_b - jsd_m
    elif loc == 'R':
        diff1 = jsd_f - jsd_b
        diff2 = jsd_f - jsd_m
    seq_df = copy.deepcopy(df)
    seq_df = seq_df[(seq_df['LOC'] == loc) & (seq_df['ORDER'] != 'a')].reset_index()
    p1, p2 = 0, 0
    orig_order = seq_df['ORDER'].values.copy()
    all_at_once = df[df['ORDER'] == 'a']
    rng = np.random.default_rng(seed)
    for _ in range(it):
        seq_df['ORDER'] = rng.permutation(orig_order) #seq_df['ORDER'].sample(frac=1).values
        # all_at_once = df[df['ORDER'] == 'a']
        test_df = pd.concat([seq_df, all_at_once])
        jsd_f, jsd_m, jsd_b = get_jsds(test_df, loc)
        if loc == 'L':
            temp_diff1 = jsd_b - jsd_f
            temp_diff2 = jsd_b - jsd_m
        elif loc == 'R':
            temp_diff1 = jsd_f - jsd_b
            temp_diff2 = jsd_f - jsd_m
        if temp_diff1 >= diff1:
            p1 += 1
        if temp_diff2 >= diff2:
            p2 += 1
    return p1/it, p2/it

def get_singular_jsd(df, loc, order):
    df_at_loc = df[df['LOC'] == loc]
    dist_a = np.array(df_at_loc[df_at_loc['ORDER'] == 'a'][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)
    dist_ord = np.array(df_at_loc[df_at_loc['ORDER'] == order][['PROP_L', 'PROP_X', 'PROP_R']].mean().values)
    jsd = jsd_no_nan(dist_ord, dist_a)
    return jsd

def perm_test2(df, loc, order, it, seed=18):
    jsd = get_singular_jsd(df, loc, order)
    seq_df = copy.deepcopy(df)
    
    seq_df = seq_df[(seq_df['LOC'] == loc) & (seq_df['ORDER'].isin(['a', order]))].reset_index()
    p = 0
    rng = np.random.default_rng(seed)
    for _ in range(it):
        seq_df['ORDER'] = seq_df['ORDER'].sample(frac=1, random_state=rng).values
        rand_jsd = get_singular_jsd(seq_df, loc, order)
        if rand_jsd >= jsd:
            p += 1
    return p/it

def format_jsd(x):
    return f"{x:.3f}"           

def format_p(p):
    return "< .001" if p < .001 else ">.99" if p > .99 else f"{p:.2f}".lstrip("0") if (p>=.05 or (.01 <= p < .045)) else f"{p:.3f}".rstrip("0").lstrip("0")  

def jsd_table_by_loc(human_df, simulated_df, loc, perms):
    rows = {}
    for source, df in [("Human", human_df), ("Simulated", simulated_df)]:
        f, m, b = get_jsds(df, loc[0])
        jsds = {"Forward": f, "Middle": m, "Backward": b}
        row = {}
        for order in ['Forward', 'Middle', 'Backward']:
            row[(order, "JSD")] = jsds[order]
            row[(order, "p")] = perm_test2(df, loc[0], order[0].lower(), perms, seed=101)
        rows[(loc, source)] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.index = pd.MultiIndex.from_tuples(table.index)
    return table

def make_jsd_table(human_df, simulated_df, perms, save=False, fname=None):
    orders = ['Forward', 'Middle', 'Backward']
    table = pd.concat([jsd_table_by_loc(human_df, simulated_df, "Left", perms), 
                       jsd_table_by_loc(human_df, simulated_df, "Right", perms)])
    jsd_cols = [(order, "JSD") for order in orders]
    num = table[jsd_cols].astype(float)      
    row_max = num.max(axis=1)            
    for order in orders:
        is_max = num[(order, "JSD")] == row_max
        table[(order, "JSD")] = [rf"\textbf{{{format_jsd(x)}}}" if m else format_jsd(x)
                                 for x, m in zip(num[(order, "JSD")], is_max)]
        table[(order, "p")] = table[(order, "p")].map(format_p)
    if save:
        latex = table.to_latex(escape=False, multicolumn_format="c", column_format="rlcccccc")
        lines = [l for l in latex.splitlines() if not l.lstrip().startswith(r"\cline")]
        h = next(i for i, l in enumerate(lines) if "Forward" in l)
        lines.insert(h + 1, r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
        r = next(i for i, l in enumerate(lines) if "Right" in l)
        lines.insert(r, r"\midrule")
        if fname is None:
            with open('Tables/JSD_tab.tex', 'w') as f:
                f.write("\n".join(lines))
        else:
            with open(F'Tables/{fname}', 'w') as f:
                f.write("\n".join(lines))          
    return table

def get_JSD_diffs(loc, human_df, simulated_df, perms, seed=13):
    if loc == 'L':
        ref = "Backward"
        others =  ["Forward", "Middle"]
    elif loc == 'R':
        ref = "Forward"
        others = ["Backward", "Middle"]
    else:
        raise(ValueError, "loc should be 'L' or 'R'")
    rows = {}
    for data, df in [("Human", human_df), ("Simulated", simulated_df)]:
        f, m, b = get_jsds(df, loc)
        jsd = {"Forward": f, "Middle": m, "Backward": b}
        p1, p2 = perm_test(df, loc, perms, seed)   
        row = {}
        for other, p in zip(others, [p1, p2]):
            label = f"{ref} - {other}"
            row[(label, "Diff")] = jsd[ref] - jsd[other]
            row[(label, "p")] = p
        rows[data] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table

def show_jsd_diffs(table):
    out = table.copy().astype(object)
    for c in table.columns.get_level_values(0).unique():
        out[(c, "Diff")] = table[(c, "Diff")].map(format_jsd)
        out[(c, "p")] = table[(c, "p")].map(format_p)
    print(out.to_string())

def build_score_row(df, dataset):
    sub_table = (df[df["LOC"] != "C"]
         .groupby(["LOC", "ORDER"])["SCORE"].mean()
         .reset_index())
    sub_table["SCORE"] -= sub_table.groupby("LOC")["SCORE"].transform("min") 
    sub_table["DATA"] = dataset
    return sub_table

def get_score_table(human_df, simulated_df, save=False, fname=None):
    order_key = {"f": "Forward", "m": "Middle", "b": "Backward", "a": "All at once"}
    scores = pd.concat([build_score_row(human_df, "Human"),
                        build_score_row(simulated_df, "Simulated")])
    table = scores.pivot_table(index=["LOC", "DATA"], columns="ORDER", values="SCORE")
    table = table.rename(index={"L": "Left", "R": "Right"}, columns=order_key)
    table = table.reindex(columns=list(order_key.values()))
    table = table.reindex([("Left", "Human"), ("Left", "Simulated"),
                        ("Right", "Human"), ("Right", "Simulated")])
    table = table.round().astype(int)
    table.index.names = [None, None]     
    table.columns.name = None
    if save:
        latex = table.to_latex(escape=False, column_format="llcccc")
        lines = [l for l in latex.splitlines() if not l.lstrip().startswith(r"\cline")]
        r = next(i for i, l in enumerate(lines) if "Right" in l)
        lines.insert(r, r"\midrule")
        if fname is None:
            with open('Tables/ckmm_score_table.tex', 'w') as f:
                f.write("\n".join(lines))
        else:
            with open(f'Tables/{fname}', 'w') as f:
                f.write("\n".join(lines))
    return table

def get_prop_table(seq_df, save=False, fname=None):
    prop_same = (
        seq_df
        .groupby(['PROP_SAME', 'FIRST'])
        .agg(
            N=('P_ID', 'size'),
        )
        .reset_index()
    )
    prop_same = (
        seq_df
        .groupby(['PROP_SAME', 'FIRST'])
        .size()
        .unstack(fill_value=0)
    )
    prop_same['Total'] = prop_same.sum(axis=1)
    prop_same['Percentage'] = (
        prop_same['Total'] / len(seq_df) * 100
    ).round(1)
    prop_same[['L', 'R', 'X']] = (
        prop_same[['L', 'R', 'X']]
        .div(prop_same['Total'], axis=0)
    ).round(3)
    prop_same = prop_same.reset_index()
    if save:
        subset = prop_same[prop_same["PROP_SAME"] >= 2/3 ][["Total", "L", "R", "X"]].copy()
        subset.columns = pd.MultiIndex.from_tuples([
            ("", "Trials"),
            ("First Item Category", "L"),
            ("First Item Category", "R"),
            ("First Item Category", "X"),
        ])
        subset.index = [f"{i} of 12" for i in subset.index]
        subset.index.name = None
        out = subset.astype(object)
        out[("", "Trials")] = subset[("", "Trials")].astype(int).astype(str)
        for c in ["L", "R", "X"]:
            out[("First Item Category", c)] = subset[("First Item Category", c)].map(lambda x: f"{x:.3f}")
        latex = out.to_latex(escape=False, multicolumn_format="c", column_format="rrccc",
                     index_names=False)

        lines = [l for l in latex.splitlines() if not l.lstrip().startswith(r"\cline")]
        h = next(i for i, l in enumerate(lines) if "First Item Category" in l)
        lines[h] = lines[h].replace("&  &", r"& Trials &", 1) if False else lines[h]
        lines.insert(h + 1, r"\cmidrule(lr){3-5}")
        if fname is None:
            with open('Tables/first_category_same_table.tex', 'w') as f:
                f.write("\n".join(lines))
        else:
            with open(f'Tables/{fname}', 'w') as f:
                f.write("\n".join(lines))
        
    return prop_same
