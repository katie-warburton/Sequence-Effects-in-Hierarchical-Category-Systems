import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def bootstrap_LXR_confidence_intervals(df, p):
    np.random.seed(13)
    avg_prop = np.empty((p, 3))
    N = df.shape[0]
    lb, ub = int(0.025*p)-1, int(0.975*p)-1
    df = df[['PROP_L', 'PROP_X', 'PROP_R']]
    for i in range(p):
        sample_df = df.sample(N, replace=True)
        avg_prop[i,:] = sample_df.mean(axis=0).to_list()
    l_vec, x_vec, r_vec = avg_prop[:,0], avg_prop[:,1], avg_prop[:,2]
    l_vec.sort(), x_vec.sort(), r_vec.sort()
    return np.array([l_vec[lb], x_vec[lb], r_vec[lb]]), np.array([l_vec[ub], x_vec[ub], r_vec[ub]])

def plot_order_effects(df, axes, trials=True):
    grouped_df = df.groupby(['LOC', 'ORDER']).mean(numeric_only=True).reset_index()
    counts = df.groupby(['LOC', 'ORDER']).count().reset_index()
    overall = df.groupby(['ORDER']).mean(numeric_only=True).reset_index()
    overall['LOC'] = ['O'] * len(overall)
    grouped_df = pd.concat([grouped_df, overall])
    orders = ['Forward', 'Middle', 'Backward', 'All at once']
    locations = ['Left', 'Centred', 'Right', 'Overall']
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=9)
            x = ['L', 'X', 'R']
            order = orders[j].lower()[0]
            loc = locations[i][0]
            y = grouped_df[(grouped_df['LOC'] == loc) & (grouped_df['ORDER'] == order)][['PROP_L', 'PROP_X', 'PROP_R']].values[0]
            if i < 3:
                ci_lb, ci_ub = bootstrap_LXR_confidence_intervals(df[(df['LOC'] == loc) & (df['ORDER'] == order)], 5000)
            else:
                ci_lb, ci_ub = bootstrap_LXR_confidence_intervals(df[df['ORDER'] == order], 5000)
            ci_lb = y - ci_lb
            ci_ub = ci_ub - y
            y_err = np.vstack([ci_lb, ci_ub])
            ax.bar(
                x, y,
                color=['#D81B60', '#1E88E5', '#FFC107'],
                edgecolor='black',
                width=0.75,
                linewidth=0.75,
                yerr=y_err,
                capsize=2,
                error_kw=dict(elinewidth=0.75, capthick=0.75)
            )
            if i == 0:
                ax.set_title(orders[j], fontsize=9.5, pad=3)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(locations[i], fontsize=10)
                ax.set_yticks([0.0, 0.5, 1.0])
            else:
                ax.set_yticks([])
            ax.set_ylim(0, 1)
            if trials:
                if i < 3:
                    t = counts[(counts['LOC'] == loc) & (counts['ORDER'] == order)]['P_ID'].iloc[0]
                else:
                    total_trials = counts.groupby('ORDER').sum().reset_index()
                    t = total_trials[total_trials['ORDER'] == order]['P_ID'].iloc[0]
                ax.text(-0.42, 0.86, f"T = {t}", size=8)

def combined_order_effects_plot(df_human, df_simulated, fname='Figures/LXR_by_order_loc_combined.jpg', figsize=(8.8, 4.8)):
    fig = plt.figure(figsize=figsize)
    outer_gs = fig.add_gridspec(
        1, 2,
        left=0.06,
        right=0.99,
        top=0.90,
        bottom=0.14,
        wspace=0.25
    )
    gs_human = outer_gs[0].subgridspec(4, 4, wspace=0.15, hspace=0.20)
    gs_simulated = outer_gs[1].subgridspec(4, 4, wspace=0.15, hspace=0.20)
    axes_human = np.empty((4, 4), dtype=object)
    axes_simulated = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            axes_human[i, j] = fig.add_subplot(gs_human[i, j])
            axes_simulated[i, j] = fig.add_subplot(gs_simulated[i, j])
    plot_order_effects(df_human, axes_human, trials=True)
    plot_order_effects(df_simulated, axes_simulated, trials=False)
    fig.text(0.01, 0.99, 'a) Human', fontsize=10, ha='left', va='top')
    fig.text(0.54, 0.99, 'b) Simulated', fontsize=10, ha='left', va='top')
    handles = [
        plt.Rectangle((0, 0), 0.5, 0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.75),
        plt.Rectangle((0, 0), 0.5, 0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.75),
        plt.Rectangle((0, 0), 0.5, 0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.75)
    ]
    fig.legend(
        handles,
        ['L', 'X', 'R'],
        loc='lower center',
        bbox_to_anchor=(0.11, 0.025),
        ncol=3,
        frameon=False,
        fontsize=9
    )
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.show()

def bootstrap_X_confidence_intervals(df, p):
    np.random.seed(13)
    has_x = np.empty((p, 2))
    N = df.shape[0]
    lb, ub = int(0.025*p)-1, int(0.975*p)-1
    df = df[['DEPTH', 'HAS_X']]
    for i in range(p):
        sample_df = df.sample(N, replace=True)
        has_x[i,:] = sample_df.groupby(['DEPTH']).mean().reset_index()['HAS_X'].to_list()
    two_vec, three_vec = has_x[:,0], has_x[:,1]
    two_vec.sort(), three_vec.sort()
    return np.array([two_vec[lb], three_vec[lb]]), np.array([two_vec[ub], three_vec[ub]])
    
def new_category_plot(df, fname='Figures/X_by_depth.jpg', figsize=(3,3)):
    df_hyp3 = df[['DEPTH', 'HAS_X']].groupby(['DEPTH']).mean().reset_index()
    ci_lb, ci_ub  = bootstrap_X_confidence_intervals(df, 5000)
    y = np.array(df_hyp3['HAS_X'].to_list())
    ci_lb = y - ci_lb
    ci_ub = ci_ub - y
    y_err = np.vstack([ci_lb, ci_ub])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x='DEPTH', height='HAS_X' , data=df_hyp3, color=['#1E88E5', '#1E88E5'],  yerr=y_err, error_kw=dict(elinewidth=0.75, capthick=0.75), edgecolor="black", capsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('% with X')
    ax.set_xlabel('Depth')
    ax.set_xticks([2, 3])
    ax.text(1.6, 0.92, f'T = {df.shape[0]}', size=10)
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'{fname}', dpi=300, bbox_inches='tight') 

def generate_summary_LXR_plot(df, column, colVals, names, figsize=(6.25, 2.15), overall=False, fname='Figures/summary.jpg', legend=True, ylabel='Mean % of items', trials=True, title=None):
    stim_df = df.groupby([column]).mean(numeric_only=True).reset_index()[[column, 'PROP_L', 'PROP_R', 'PROP_X']]   
    counts = df.groupby([column]).count().reset_index()[[column, 'PROP_L', 'PROP_R', 'PROP_X']]   
    numAx = len(colVals)
    if overall:
        avg = df[['PROP_L', 'PROP_X', 'PROP_R']].mean(numeric_only=True).values
        numAx += 1
    fig, axes = plt.subplots(1, numAx, figsize=figsize, constrained_layout=True)
    for i in range(numAx):
            ax = axes[i]
            x = ['L', 'X', 'R']
            if overall and i == 0:
                y = avg
                if trials:
                    ax.text(-0.42, 0.87, f'T = {df.shape[0]}', size=8)
                ax.set_title('Overall', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df, 5000)
            elif overall:
                val = colVals[i-1]
                y = stim_df[stim_df[column] == val][['PROP_L', 'PROP_X', 'PROP_R']].values[0]
                if trials: 
                    ax.text(-0.42, 0.87, f"T = {counts[counts[column] == val]['PROP_L'].iloc[0]}", size=8)
                ax.set_title(f'{names[i-1]}', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df[df[column] == val], 5000)
            else:
                val = colVals[i]
                y = stim_df[stim_df[column] == val][['PROP_L', 'PROP_X', 'PROP_R']].values[0]
                if trials:
                    ax.text(-0.42, 0.87, f"T = {counts[counts[column] == val]['PROP_L'].iloc[0]}", size=8)
                ax.set_title(f'{names[i]}', fontsize=9.5)
                ci_lb, ci_ub  = bootstrap_LXR_confidence_intervals(df[df[column] == val], 5000)
            ci_lb = y - ci_lb
            ci_ub = ci_ub - y
            y_err = np.vstack([ci_lb, ci_ub])
            ax.bar(x, y, color=['#D81B60', '#1E88E5', '#FFC107'], edgecolor='black', width=0.75, linewidth=0.75, yerr=y_err, capsize=2,  error_kw=dict(elinewidth=0.75, capthick=0.75))
            ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=8)
            if i == 0:
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_ylabel(ylabel, fontsize=8)
            else:
                ax.set_yticks([])
            ax.set_ylim(0, 1)
            if title is not None:
                fig.canvas.draw()
                ylabel_bbox = axes[0].yaxis.get_label().get_window_extent()
                fig_left_coord = fig.transFigure.inverted().transform([ylabel_bbox.x0, 0])[0]
                fig.suptitle(f'{title}',  x=fig_left_coord, ha="left", fontsize=9.5)

            if legend:
                handles = [
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#D81B60', edgecolor='black', linewidth=0.75),
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#1E88E5', edgecolor='black', linewidth=0.75),
                    plt.Rectangle((0,0),0.5,0.5, facecolor='#FFC107', edgecolor='black', linewidth=0.75),
                ]
                labels = ['L', 'X', 'R']
                fig.legend(
                    handles,
                    labels,
                    loc='lower left',
                    bbox_transform=fig.transFigure,
                    bbox_to_anchor=(-0.02, -0.15),
                    ncol=3,
                    frameon=False,
                    fontsize=8
                )
    fig.savefig(f'{fname}', dpi=600, bbox_inches='tight')

def same_as_dist_combined(human_seq, simulated_seq, fname='Figures/same_as_dist_combined.jpg', figsize=(8, 2), include_distractors=False):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    same_as_dist(axes[0], human_seq, trials=True, include_distractors=include_distractors)
    same_as_dist(axes[1], simulated_seq, trials=False, include_distractors=include_distractors)
    fig.text(0.055, 0.99, 'a) Human', fontsize=11, ha='left', va='top')
    fig.text(0.495, 0.99, 'b) Simulated', fontsize=11, ha='left', va='top')
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(f'{fname}', dpi=600, bbox_inches='tight')

def same_as_dist(ax, seq_df, include_distractors=False, trials=True):
    dist = seq_df['PROP_SAME'].value_counts().reset_index()
    dist['prop'] = dist['count'] / seq_df.shape[0]
    labels =  dist['PROP_SAME'].to_list()
    labels.sort()
    ax.bar(x='PROP_SAME', height='prop', data=dist,  color='#004D40', edgecolor="black", width=1/(dist.shape[0]))
    if include_distractors:
        ax.set_xticks(labels, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    else:
        ax.set_xticks(labels, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    if trials:
        ax.text(-0.05, 0.46, f'T = {seq_df.shape[0]}', fontsize=11)
    ax.set_ylabel('% of Trials (T)', fontsize=11)
    ax.set_xlabel('Number of Items', fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines[['right', 'top']].set_visible(False)