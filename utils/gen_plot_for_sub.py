# Includes the code to generate plot for submission
import os
import pdb
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib as mpl
from sklearn.metrics import ConfusionMatrixDisplay
from helper_utils.helper_methods import list_hardcode_datasets_and_their_splits
from helper_utils.color_helper import get_continuous_cmap
from constants import MODEL_NICE_NAMES, PRETRAINED_MODEL, UNPRETRAINED_MODEL, DATASET_NICE_NAMES, SPLIT_NICE_NAMES, default_dataset_mapping, all_dataset_mapping, default_model_names, lexical_dataset_mapping, length_dataset_mapping, all_exclude_length_dataset_mapping, SYNTHETIC_DATA, NATURAL_DATA, dataset_color_mapping, raw_dataset_mapping, lexical_without_orig_dataset_mapping, lexical_w_orig_mapping
from analysis_utils import get_eval_difference


def concurrence_conf_matrix(dataset_and_splits=default_dataset_mapping, coref='Kendall'):
    """
    Figure 1 and the largest figure in appendix
    Return a confusion matrix, out level label, and inner level label
    """
    # Load Concurrence
    # Load the concurrence table
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    # Loop through dataset mapping, create the inner label (xtick) list and 
    raw_label_list = []
    inner_temp_label_list = []
    inner_label_list = []
    ticks = []
    minor_ticks = []
    tick_num = 0
    prev_large_tick = -2
    tick_locs = [13.05, 7.05, 4.05, 1.05]
    for dataset in dataset_and_splits:
        #--
        inner_label_list.append(DATASET_NICE_NAMES[dataset])
        raw_label_list.append(dataset)
        tick_loc = tick_num + 1.02
        if tick_loc - prev_large_tick <= 2:
            tick_loc += 2.0
        # tick_loc = tick_locs.pop()
        ticks.append(tick_loc)
        prev_large_tick = tick_loc
        #--
        mid_point = int(len(dataset_and_splits[dataset]) / 2) + tick_num - 1
        for split in dataset_and_splits[dataset]:
            # if tick_num == mid_point:
            #     inner_label_list.append(DATASET_NICE_NAMES[dataset])
            #     raw_label_list.append(dataset)
            #     if len(dataset_and_splits[dataset]) % 2 == 0:
            #         # ticks.append(tick_num + 0.02)
            #         ticks.append(tick_num - 0.03)
            #     else:
            #         ticks.append(tick_num + 0.01)
            inner_label_list.append(SPLIT_NICE_NAMES[split])
            ticks.append(tick_num)
            inner_temp_label_list.append(dataset + '-' + split)
            raw_label_list.append(dataset + '-' + split)
            tick_num += 1
        # inner_label_list.append('\n' + dataset)
    # Construct the confuson matrix
    concurrence_matrix = np.zeros([len(inner_temp_label_list), len(inner_temp_label_list)])
    for x_idx, setupx in enumerate(inner_temp_label_list):
        for y_idx, setupy in enumerate(inner_temp_label_list):
            if '-' in setupx and '-' in setupy:
                # Find the corresponding values
                dataset1 = setupx.split('-')[0]
                split1 = setupx.split('-')[1]
                try:
                    eval_split1 = 'test' if dataset1 != 'COGS' or split1 == 'length' else setupx.split('-')[2]
                except:
                    pdb.set_trace()
                dataset2 = setupy.split('-')[0]
                split2 = setupy.split('-')[1]
                eval_split2 = 'test' if dataset2 != 'COGS' or split2 == 'length' else setupy.split('-')[2]
                current_concur = concurrences.loc[(concurrences["Dataset1"] == dataset1) & (concurrences["Dataset2"] == dataset2) & (concurrences["Split1"] == split1) & (concurrences["Split2"] == split2) & (concurrences["EvalSplit1"] == eval_split1) & (concurrences["EvalSplit2"] == eval_split2)]

                if len(current_concur) == 1:
                    concurrence_matrix[x_idx, y_idx] = current_concur['concurrence'].item()
                else:
                    # pdb.set_trace()
                    concurrence_matrix[x_idx, y_idx] = -1.5
    # Plotting
    mpl.rc('xtick', labelsize=11)
    mpl.rc('ytick', labelsize=11)

    max_concur = concurrence_matrix.max()
    min_concur = concurrence_matrix.min()
    cticks, cticks_labels = helper_construct_colorbarticks(min_concur, max_concur)

    fig, ax = plt.subplots()
    divnorm = mcolors.TwoSlopeNorm(vmin=-1,vcenter=0, vmax=1)
    cmap_hex_list = ['#d15438', '#ffffff', '#0277a1']
    # cmap_hex_list = ['#9a634f', '#ffffff', '#6d6a8c']
    im = ax.imshow(concurrence_matrix, cmap=get_continuous_cmap(cmap_hex_list), norm=divnorm)
    # im = ax.imshow(concurrence_matrix, vmin=-1, vmax=1, cmap='twilight')

    ax.set_xticks(ticks, rotation='vertical')
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xticklabels(inner_label_list, rotation='vertical')
    ax.set_yticklabels(inner_label_list)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    cbar = fig.colorbar(im, ticks=cticks)
    cbar.ax.set_yticklabels(cticks_labels)
    # Change to set_params for all these setting above, if I have extra time.
    for xticklabel, yticklabel, full_label in zip(plt.gca().get_xticklabels(), plt.gca().get_yticklabels(), raw_label_list):
        tickcolor = dataset_color_mapping[full_label.split('-')[0]]      
        if '-' not in full_label:
            xticklabel.set_y(+1.18)
            # Uncomment below to generate table 7
            # xticklabel.set_y(+1.23)
            xticklabel.set_fontsize(12)
            xticklabel.set_rotation(0)
            yticklabel.set_rotation(90)
            yticklabel.set_x(-0.18)
            # yticklabel.set_x(-0.23)
            yticklabel.set_fontsize(12)
        # Uncomment below to generate table 7
        # else:
        #     xticklabel.set_fontsize(10)
        #     yticklabel.set_fontsize(10)
        xticklabel.set_color(tickcolor)
        yticklabel.set_color(tickcolor)
    # Add legend with average values
    # legend_elements = [Line2D([0], [0], color='#FFFFFF', lw=0.1, label='Spider=0.30'),
    #                Line2D([0], [0], color='#FFFFFF', lw=0.1, label='COGS=0.36'),
    #                Line2D([0], [0], color='#FFFFFF', lw=0.1, label='GeoQuery=0.29'),
    #                Line2D([0], [0], color='#FFFFFF', lw=0.1, label='SCAN=0.15')]
    # # labels = ['Spider', 'COGS', 'GeoQuery', 'SCAN']
    # ax.legend(handles=legend_elements, ncol=5, title='Average:', bbox_to_anchor=(1.0, -0.01), handletextpad=0.001,  columnspacing=0.001)
    # Uncomment below line to generate figure 1, otherwise generate figure 7
    ax.legend(title='Average: NACS=0.16, Spider=0.25, COGS=0.36, GeoQuery=0.27, SCAN=0.15', bbox_to_anchor=(1.2, -0.01), fontsize=10)
    plt.savefig(os.environ['BASE_DIR'] + f'/results/conf_matrix.png', dpi=300, bbox_inches = "tight")
    plt.savefig(os.environ['BASE_DIR'] + f'/results/conf_matrix.pdf', bbox_inches = "tight")

    return concurrence_matrix

def gen_perf_plot_w_std(dataset_and_splits=default_dataset_mapping, model_names=default_model_names, metric_name='ignore_space'):
    """
    Generate the overall performance table (table 1) with standard deviation
    Column names = | Dataset | Split | LSTM Uni | LSTM Bi | Transformer | T5 | BART |
    """
    
    # Load the performance table
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')

    # Create row header
    rows = []
    row_header = ['Dataset', 'Split']
    for model in model_names:
        row_header.append(MODEL_NICE_NAMES[model])
        row_header.append(MODEL_NICE_NAMES[model])
    row_header.append('Avg')
    # Loop through the dataset and splits
    for dataset in dataset_and_splits:
        for split in dataset_and_splits[dataset]:
            curr_row = [DATASET_NICE_NAMES[dataset], SPLIT_NICE_NAMES[split]]
            curr_avg = 0
            for model in model_names:
                # Loop through the models, retrieve the corresponding performance
                if '-' in split:
                    data_split = split.split("-")[0]
                    eval_split = split.split("-")[1]
                    curr_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == data_split) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == eval_split)]
                else:
                    curr_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == split) & (perf_table['Model'] == model)]
                if len(curr_perf) < 1:
                    curr_row.append('IP')
                elif str(curr_perf['Std'].item()) == 'nan':
                    # TODO: Remove this clause when we have all performance for all models
                    curr_row.append(str(round(curr_perf[metric_name].item() * 100.0, 1)) + "(± IP)")
                else:
                    curr_avg += round(curr_perf[metric_name].item() * 100.0, 1)
                    if curr_perf['Std'].item() == 0:
                        curr_row.append(str(round(curr_perf[metric_name].item() * 100.0, 1)))
                        curr_row.append("±" + str(curr_perf['Std'].item())[1:])
                    else:
                        # curr_row.append(str(round(curr_perf[metric_name].item() * 100.0, 1)) + " (±" + str(curr_perf['Std'].item())[1:] + ")")
                        curr_row.append(str(round(curr_perf[metric_name].item() * 100.0, 1)))
                        curr_row.append("±" + str(curr_perf['Std'].item())[1:])
            curr_avg /= len(model_names)
            curr_row.append(round(curr_avg, 1))
            rows.append(curr_row)
    return pd.DataFrame(rows, columns=row_header)

def gen_concurrence_table(dataset_and_splits=default_dataset_mapping, coref='Kendall', filter_type="top 5%"):
    """
    Show the highest and/or lowest set of concurrence values. This is used to generate Table 2, 3, and 4ab
    filter_type={'0.8', 'top 5%', 'last 5%', 'all', 'lexical split', 'length split'}
    """
    # Load the concurrence table
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    setup_to_show = helper_gen_data_split_str(dataset_and_splits)
    concurrences = helper_clean_concurrence_table(concurrences, setup_to_show)
    # Sort the concurrence values
    if filter_type == 'last 5%':
        concurrences = concurrences.sort_values(by=['concurrence'], ascending=True)
    else:
        concurrences = concurrences.sort_values(by=['concurrence'], ascending=False)
    # Take the ones higher than 0.8 or take top 5
    header_row = ["Dataset A", "Split A", "Dataset B", "Split B", "Concur"]
    rows = []

    count = len(concurrences) * 0.05
    print(f"Total pairs = {len(concurrences)}, Display pairs = {count}")
    # pdb.set_trace()
    for idx, concur_row in concurrences.iterrows():
        if (filter_type == 'last 5%' or  filter_type == 'top 5%') and count > 0 or filter_type == '0.8' and (concur_row['concurrence'] >= 0.8 or concur_row['concurrence'] <= -0.6) or filter_type == '0.7' and concur_row['concurrence'] >= 0.7 or filter_type == 'all':
            if concur_row["Dataset1"] + '-' + concur_row["Split1"] in setup_to_show and concur_row["Dataset2"] + '-' + concur_row["Split2"] in setup_to_show:
                # Add to rows
                rows.append([DATASET_NICE_NAMES[concur_row["Dataset1"]],
                                SPLIT_NICE_NAMES[concur_row["Split1"]], 
                                DATASET_NICE_NAMES[concur_row["Dataset2"]],
                                SPLIT_NICE_NAMES[concur_row["Split2"]], 
                                round(concur_row["concurrence"], 2)])
                count -= 1
        elif filter_type == 'lexical split' and concur_row["Dataset1"] == concur_row["Dataset2"]:
            if concur_row["Split1"] in default_dataset_mapping[concur_row["Dataset1"]] and concur_row["Split2"] not in default_dataset_mapping[concur_row["Dataset2"]] or concur_row["Split1"] not in default_dataset_mapping[concur_row["Dataset1"]] and concur_row["Split2"] in default_dataset_mapping[concur_row["Dataset2"]] or concur_row["Dataset2"] == "COGS":
                # Only compute the original split vs the modified split
                rows.append([DATASET_NICE_NAMES[concur_row["Dataset1"]],
                                    SPLIT_NICE_NAMES[concur_row["Split1"]], 
                                    DATASET_NICE_NAMES[concur_row["Dataset2"]],
                                    SPLIT_NICE_NAMES[concur_row["Split2"]], 
                                    round(concur_row["concurrence"], 2)])
                count -= 1
            
    return pd.DataFrame(rows, columns=header_row)

def plot_perf_vs_concur(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    """
    A scatter plot of performance on one dataset (averaged across models) / avg of the paired performance v.s. Concurrence - shape determine by the pairs type (syn-syn, nat-nat, syn-nat))
    Figure 6
    """
    # Load performance and concurrence
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')

    # res will become a dataframe with cnocur | perf | type
    res = []
    res_self = []
    for dataset1 in dataset_mapping:
        for dataset2 in dataset_mapping:
            eval_split1 = 'gen' if dataset1 == 'COGS' else 'test'
            eval_split2 = 'gen' if dataset2 == 'COGS' else 'test'
            # Loop through each split pair
            for split1 in dataset_mapping[dataset1]:
                if '-' in split1:
                    split1 = split1.split('-')[0]
                for split2 in dataset_mapping[dataset2]:
                    if '-' in split2:
                        split2 = split2.split('-')[0]
                    # if dataset1 != dataset2 or split1 != split2:
                    # Compute average performance across models and datasets
                    setup1_perfs = perf_table.loc[(perf_table['Dataset'] == dataset1) & (perf_table['Split'] == split1) & (perf_table['Eval Split'] == eval_split1)]
                    setup2_perfs = perf_table.loc[(perf_table['Dataset'] == dataset2) & (perf_table['Split'] == split2) & (perf_table['Eval Split'] == eval_split2)]
                    current_concur = concurrences.loc[ (concurrences['Dataset1'] == dataset1) & (concurrences['Dataset2'] == dataset2) & (concurrences['Split1'] == split1) &(concurrences['Split2'] == split2) & (concurrences['EvalSplit1'] == eval_split1) & (concurrences['EvalSplit2'] == eval_split2)]
                    # Get pair types
                    if dataset1 in SYNTHETIC_DATA and dataset2 in SYNTHETIC_DATA:
                        pair_type = 'syn-syn'
                    elif dataset1 in NATURAL_DATA and dataset2 in NATURAL_DATA:
                        pair_type = 'nat-nat'
                    else:
                        pair_type = 'nat-syn'
                    if len(setup1_perfs) >= 1 and len(setup2_perfs) >= 1 and len(current_concur) == 1:
                            if dataset1 != dataset2 or split1 != split2:
                                res.append({'concur': current_concur['concurrence'].item(),
                                        'perf': round((setup2_perfs['ignore_space'].mean() + setup1_perfs['ignore_space'].mean()) / 2 * 100, 2),
                                        'pair_type': pair_type
                                            })
                            else:
                                # Record it and make it into different shapes of dots
                                res_self.append({'concur': current_concur['concurrence'].item(),
                                        'perf': round((setup2_perfs['ignore_space'].mean() + setup1_perfs['ignore_space'].mean()) / 2 * 100, 2),
                                        'pair_type': pair_type
                                            })    
    res = pd.DataFrame(res)
    res_self = pd.DataFrame(res_self)

    min_val = min(res['concur'].min(), res_self['concur'].min())
    # Start plotting
    # syn-syn
    print(res.to_string())
    mix_res = res.loc[res['pair_type'] == 'nat-syn']
    plt.scatter(mix_res['perf'].tolist(), mix_res['concur'].tolist(), c='#6ea5f7', alpha=0.5, label='Natural-Synthetic', s=28)
    syn_res = res.loc[res['pair_type'] == 'syn-syn']
    plt.scatter(syn_res['perf'].tolist(), syn_res['concur'].tolist(), c='#c97798', alpha=0.5, label='Synthetic-Synthetic', s=28)
    nat_res = res.loc[res['pair_type'] == 'nat-nat']
    plt.scatter(nat_res['perf'].tolist(), nat_res['concur'].tolist(), c='#7db338', alpha=0.5, label='Natural-Natural', s=28)
    avg_values = dict()
    avg_values['syn-syn'] = syn_res['concur'].sum() / len(syn_res['concur'])
    avg_values['nat-syn'] = mix_res['concur'].sum() / len(mix_res['concur'])
    avg_values['nat-nat'] = nat_res['concur'].sum() / len(nat_res['concur'])
    print(avg_values)

    syn_res_self = res_self.loc[res['pair_type'] == 'syn-syn']
    plt.scatter(syn_res_self['perf'].tolist(), syn_res_self['concur'].tolist(), c='#c97798', alpha=0.5, marker='^', label='Self-Synthetic', s=30)
    nat_res_self = res_self.loc[res_self['pair_type'] == 'nat-nat']
    plt.scatter(nat_res_self['perf'].tolist(), nat_res_self['concur'].tolist(), c='#7db338', alpha=0.5, marker='v', label='Self-Natural', s=30)
    
    plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.3)
    xticks = [x * 10 for x in range(11)]
    yticks = [x * 0.1 for x in range(int(min_val * 10) - 1, 11, 2)]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.legend(loc='lower right', fontsize=11)
    # plt.legend(fontsize=11)
    plt.xlabel('Average EM between two splits in the pair', fontsize=16)
    plt.ylabel('Concurrence', fontsize=17)
    # plt.title(f'Performance-Concurrence Alignment')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/perf_vs_concur.png')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/perf_vs_concur.pdf')
    return res

def plot_density_concurrence_count_by_type(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    """
    Figure 4
    Bar plot the count of concurrence by the type of pairs.
    """
    # Load Concurrence
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    # res will become a dataframe with cnocur | perf | type
    res = {
        'DiffData, SameSplit' : [],
        'SameData, DiffSplit' : [],
        'DiffData, DiffSplit' : [],
        'SameData, SameSplit' : []
           }
    for dataset1 in dataset_mapping:
        for dataset2 in dataset_mapping:
            eval_split1 = 'gen' if dataset1 == 'COGS' else 'test'
            eval_split2 = 'gen' if dataset2 == 'COGS' else 'test'
            # Loop through each split pair
            for split1 in dataset_mapping[dataset1]:
                for split2 in dataset_mapping[dataset2]:
                    current_concur = concurrences.loc[ (concurrences['Dataset1'] == dataset1) & (concurrences['Dataset2'] == dataset2) & (concurrences['Split1'] == split1) &(concurrences['Split2'] == split2) & (concurrences['EvalSplit1'] == eval_split1) & (concurrences['EvalSplit2'] == eval_split2)]
                    # Get pair types
                    if dataset1 != dataset2 and split1 == split2:
                        pair_type = 'DiffData, SameSplit'
                    elif dataset1 == dataset2 and split1 != split2:
                        pair_type = 'SameData, DiffSplit'
                    elif dataset1 != dataset2 and split1 != split2:
                        pair_type = 'DiffData, DiffSplit'
                    else:
                        pair_type = 'SameData, SameSplit'
                    if len(current_concur) == 1 and not current_concur['concurrence'].isnull().values.any():
                        
                            current_concur = current_concur['concurrence'].item()
                            # Add the count to the dictonary
                            if current_concur is not None:
                                res[pair_type].append(current_concur)
    plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.3)
    colors = ["#96595A", "#DA897C", "#E4E4B2", "#B2E4CF", "#0D6A82"]
    for pair_type in res:
        # Compute avg values
        # pdb.set_trace()
        avg = round(sum(res[pair_type]) / len(res[pair_type]), 2)
        sns.distplot(res[pair_type], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = pair_type + ', avg=' + str(avg), color=colors.pop())
    plt.legend(fontsize=14)
    plt.xlim([-1, 1])
    plt.xlabel("Concurrence Values", fontsize=15)
    plt.ylabel("Amount of Pairs over All Instances", fontsize=15)
    plt.savefig(os.environ['BASE_DIR'] + f'/results/density_by_type.png', bbox_inches = "tight")
    plt.savefig(os.environ['BASE_DIR'] + f'/results/density_by_type.pdf', bbox_inches = "tight")


def plot_lexical_perf_pretrainednonpretrained():
    """
    Plot a figure like figure 1 in Liu et al, in which the x-axis is an dataset EM,
    and the y-axis is another dataset EM. Better to use concurrence values to determine
    pair to plot and show.
    Figure 5

    Will count each random seed trial as one variant
    """
    # Load the performance table
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/exact_match.csv')

    # For each model, record the corresponding performance in pair1 and pair2 as lists
    model_list = list(set(perf_table['Model'])) 
    std_perfs =[]
    std_non_pretrained = []
    std_pretrained = []
    lex_perfs = []
    lex_non_pretrained = []
    lex_pretrained = []
    for model in model_list:
        for dataset in lexical_w_orig_mapping:
            for standard_split in lexical_w_orig_mapping[dataset]:
                std_split_name = standard_split.split('-')[0] if '-' in standard_split else standard_split
                std_eval_split = standard_split.split('-')[1] if '-' in standard_split else 'test'
                # Gather performance of standard split
                standard_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == std_split_name) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == std_eval_split) & (perf_table['Seed'] != 'Standard Deviation') & (perf_table['Seed'] != 'Mean')]
                for lexical_split in lexical_w_orig_mapping[dataset][standard_split]:
                    lex_split_name = lexical_split.split('-')[0] if '-' in lexical_split else lexical_split
                    lex_eval_split = lexical_split.split('-')[1] if '-' in lexical_split else 'test'
                    # Filter the rows that satisfy the requriement
                    lexical_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == lex_split_name) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == lex_eval_split) & (perf_table['Seed'] != 'Standard Deviation') & (perf_table['Seed'] != 'Mean')]
                    
                    if len(standard_perf['ignore_space']) == len(lexical_perf['ignore_space']):
                        # When we have the nonzero performance of both pairs, record
                        std_perfs += standard_perf['ignore_space'].values.flatten().tolist()
                        lex_perfs += lexical_perf['ignore_space'].values.flatten().tolist()

                        # Record to pretrained and nonpretrained
                        if model in PRETRAINED_MODEL:
                            std_pretrained += standard_perf['ignore_space'].values.flatten().tolist()
                            lex_pretrained += lexical_perf['ignore_space'].values.flatten().tolist()
                        elif model in UNPRETRAINED_MODEL:
                            std_non_pretrained += standard_perf['ignore_space'].values.flatten().tolist()
                            lex_non_pretrained += lexical_perf['ignore_space'].values.flatten().tolist()
    # Plotting
    # axs[x_idx, y_idx].plot(pair1_pretrained + pair1_non_pretrained, np.poly1d(np.polyfit(pair1_pretrained + pair1_non_pretrained, pair2_pretrained + pair2_non_pretrained, 1))(pair1_pretrained + pair1_non_pretrained), c='#525252')
    # plt.scatter(pair1_perfs, pair2_perfs, c='#40B0A6', alpha=0.5)
    plt.scatter(std_pretrained, lex_pretrained, c='#d15438', alpha=0.6, label='Pretrained Models', s=40)
    plt.scatter(std_non_pretrained, lex_non_pretrained, c='#0277a1', alpha=0.6, label='Non-pretrained Models', s=40)
    plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4)
    ticks = [x * 0.1 for x in range(11)]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend(fontsize=14, loc='upper left')
    plt.xlabel('Performance on original splits', fontsize=16)
    plt.ylabel('Performance on lexically modified splits', fontsize=17)
        
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/std_vs_lex.png')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/std_vs_lex.pdf')

def plot_perf_synnat_pretrainednonpretrained(x_setups = ['COGS-no_mod'], y_setups = ['geoquery-standard'], save_name='', coref='Kendall'):
    """
    Plot a figure like figure 1 in Liu et al, in which the x-axis is an dataset EM,
    and the y-axis is another dataset EM. Better to use concurrence values to determine
    pair to plot and show.
    Figure 3

    Will count each random seed trial as one variant
    """
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')

    fig, axs = plt.subplots(2, 2)
    ticks = [x * 0.1 for x in range(0, 12, 2)]
    idx = 0
    for x_setup, y_setup in zip(x_setups, y_setups):
        # for y_idx, y_setup in enumerate(y_setups):
                x_idx = int(idx / 2)
                y_idx = idx % 2
                idx += 1
                x_dataset = x_setup.split('-')[0]
                y_dataset = y_setup.split('-')[0]
                x_split = x_setup.split('-')[1]
                y_split = y_setup.split('-')[1]
                # Gather the list and concat
                pair1_pretrained, pair2_pretrained, pair1_non_pretrained, pair2_non_pretrained = plot_performance_pretrainednonpretrained(x_dataset, x_split, y_dataset, y_split)
                # Make the subplot
                axs[x_idx, y_idx].scatter(pair1_pretrained, pair2_pretrained, c='#d15438', alpha=0.7, label='Pretrained Models', s=23)
                axs[x_idx, y_idx].scatter(pair1_non_pretrained, pair2_non_pretrained, c='#0277a1', alpha=0.7, label='Non-pretrained Models', s=23)
                axs[x_idx, y_idx].grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4)
                axs[x_idx, y_idx].plot(pair1_pretrained + pair1_non_pretrained, np.poly1d(np.polyfit(pair1_pretrained + pair1_non_pretrained, pair2_pretrained + pair2_non_pretrained, 1))(pair1_pretrained + pair1_non_pretrained), c='#525252')
                # Get the xticks and yticks
                xticks = [round(x * 0.1, 1) for x in range(max(int(min(min(pair1_pretrained), min(pair1_non_pretrained)) * 10) - 1, 0), int(max(max(pair1_pretrained), max(pair1_non_pretrained)) * 10) + 1)]
                yticks = [round(x * 0.1, 1) for x in range(max(int(min(min(pair2_pretrained), min(pair2_non_pretrained)) * 10) - 1, 0), int(max(max(pair2_pretrained), max(pair2_non_pretrained)) * 10) + 1)]
                if len(xticks) >= 8:
                    xticks = [round(x * 0.1, 1) for x in range(max(int(min(min(pair1_pretrained), min(pair1_non_pretrained)) * 10) - 1, 0), int(max(max(pair1_pretrained), max(pair1_non_pretrained)) * 10) + 2, 2)]
                if len(yticks) >= 8:
                    yticks = [round(x * 0.1, 1) for x in range(max(int(min(min(pair2_pretrained), min(pair2_non_pretrained)) * 10) - 1, 0), int(max(max(pair2_pretrained), max(pair2_non_pretrained)) * 10) + 2, 2)]
                axs[x_idx, y_idx].set_xticks(xticks)
                axs[x_idx, y_idx].set_xticklabels(xticks, fontsize=11)
                axs[x_idx, y_idx].set_xlabel(DATASET_NICE_NAMES[x_dataset] + ' - ' + SPLIT_NICE_NAMES[x_split], fontsize=14)
                axs[x_idx, y_idx].set_yticks(yticks)
                axs[x_idx, y_idx].set_ylabel(DATASET_NICE_NAMES[y_dataset] + ' - ' + SPLIT_NICE_NAMES[y_split], fontsize=14)
                axs[x_idx, y_idx].set_yticklabels(yticks, fontsize=11)
                # Find the corresponding concurrence
                current_concur = concurrences.loc[(concurrences["Dataset1"] == x_dataset) & (concurrences["Dataset2"] == y_dataset) & (concurrences["Split1"] == x_split) & (concurrences["Split2"] == y_split)]

                if len(current_concur) == 1:
                    concur = current_concur['concurrence'].item()
                else:
                    concur = current_concur['concurrence'].loc[(concurrences["EvalSplit1"] == 'gen') | (concurrences["EvalSplit2"] == 'gen')].item()
                axs[x_idx, y_idx].legend([], (), title='τ = ' + str(round(concur, 2)))

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/pretrained_nonpretrained.png', bbox_inches="tight")
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/pretrained_nonpretrained.pdf', bbox_inches="tight")
    # Plotting
    # plt.scatter(pair1_perfs, pair2_perfs, c='#40B0A6', alpha=0.5)
    # plt.scatter(syn_pretrained, nat_pretrained, c='#40B0A6', alpha=0.5, label='Pretrained Models')
    # plt.scatter(syn_non_pretrained, nat_non_pretrained, c='#7570B3', alpha=0.5, label='Non-pretrained Models')
    # plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4)
    # ticks = [x * 0.1 for x in range(11)]
    # plt.xticks(ticks)
    # plt.yticks(ticks)
    # plt.legend()
    # # TODO: Add legend of tau and gamma for the plotted dataset
    # plt.xlabel('Performance on Synthetic Dataset')
    # plt.ylabel('Performance on Natural Dataset')
    # # plt.title('Pretrain')
    # plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/syn_nat_predictive{save_name}.png')

def plot_nice_pubcount():
    data = np.array([
        [1, 8, 40, 7],
        [4, 16, 136, 42],
        [6, 63, 37, 29],
        [44, 18, 1, 19],
        [10, 23, 15, 9],
        [3, 8, 71, 10]
    ])
    y_labels = ['Across Language', 'Across Domain', 'Robustness', 'Compositional', 'Structural', 'Across Task']
    x_labels = ['FullyGenerated', 'GeneratedShifts', 'NaturalShifts', 'PartitionedNatural']
    fig, ax = plt.subplots()
    cmap_hex_list = ['#ffffff', '#0277a1']
    # cmap_hex_list = ['#9a634f', '#ffffff', '#6d6a8c']
    im = ax.imshow(data, vmin=0, vmax=100, cmap=get_continuous_cmap(cmap_hex_list), aspect=0.7)
    # im = ax.imshow(data, vmin=0, vmax=100, cmap=get_continuous_cmap(cmap_hex_list), aspect=1.5)
    ax.set_xticks([0, 1, 2, 3], rotation='vertical')
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(x_labels, rotation='15', fontsize=15)
    ax.set_yticklabels(y_labels, fontsize=15)
    ax.set_xlabel('Shift Source', fontsize=17)
    ax.set_ylabel('Generalization Type', fontsize=16)
    for (j,i),label in np.ndenumerate(data):
        ax.text(i,j,label,ha='center',va='center', fontsize=15)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/pub_count.png')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/pub_count.pdf')

#### Beginning of Functons that generate figures/tables in the appendix
def gen_large_concurrence_table_appendix(coref='Kendall'):
    """Show concurrence values; Will be put into appendix"""
    # Load the concurrence table
    setup_to_show = helper_gen_data_split_str(all_dataset_mapping)
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    concurrences = helper_clean_concurrence_table(concurrences, setup_to_show=setup_to_show)
    # Sort the concurrence values
    concurrences = concurrences.sort_values(by=['concurrence'], ascending=False)

    header_row = ["Dataset A", "Split A", "Dataset B", "Split B", "Concur"]
    rows = []
    for _, concur_row in concurrences.iterrows():
        if concur_row['concurrence'] <= 1.0 and concur_row['concurrence'] >= -1.0:
            # Add to rows
            rows.append([DATASET_NICE_NAMES[concur_row["Dataset1"]],
                            SPLIT_NICE_NAMES[concur_row["Split1"]], 
                            DATASET_NICE_NAMES[concur_row["Dataset2"]],
                            SPLIT_NICE_NAMES[concur_row["Split2"]], 
                            round(concur_row["concurrence"], 2)])

    return pd.DataFrame(rows, columns=header_row)

def get_spider_perf_appendix(model_names=default_model_names):
    """
    Generate the spider performance to be displayed in appendix section
    """
    res = []
    # model_names = ['t5-base']
    row_header = ['Split'] + [MODEL_NICE_NAMES[x] for x in model_names]
    for split in default_dataset_mapping['spider']:
        row = [SPLIT_NICE_NAMES[split]]
        for model in model_names:
            avg = 0.0
            seeds = [0, 42, 12345] if model in ['t5-base', 'bart-base', 'transformer', 'btg'] else [0, 1, 5, 42, 12345]
            seeds = [0, 42, 12345]
            
            for seed in seeds:
                file_name = model + '_' + str(seed) + '.txt'
                if 'lstm' in model or 'uni' in model or 'bi' in model:
                    model = model.split('_')[-1]
                    file_name = model + '_' + str(seed) + '.txt.txt'
                if 'transformer' in model:
                    file_name = model + '_' + str(seed) + '.txt.txt'
                # get corresponding value, and take average
                avg += float(open(os.environ['BASE_DIR'] + '/results/spider_res/' + split + '/' + file_name).read())
            row.append(round(avg / len(seeds) * 100, 1))
        res.append(row)
    return pd.DataFrame(res, columns=row_header)

def gen_eval_difference_display(raw_table, dataset_mapping=default_dataset_mapping):
    """
    Generate the perf difference display
    Appendix Table 9
    """
    setup_to_show = helper_gen_data_split_str(dataset_mapping)
    res_rows = []
    for _, row in raw_table.iterrows():
        if row['Dataset'] + '-' + row['Split'] in setup_to_show:
            res_rows.append({
                'Dataset': row['Dataset'],
                'Split': row['Split'],
                'Model': row['Model'],
                'Eval Split': row['Eval Split'],
                'Diff': row['Diff']
            })
    return pd.DataFrame(res_rows)


#### Beginning of the Functions that generate figures/tables that are not included in paper
def gen_perf_plot(dataset_and_splits=default_dataset_mapping, model_names=default_model_names, metric_name='ignore_space', raw_table=None):
    """
    Not generating the figure to be presented. A legacy code to generate raw performance
    Column names = | Dataset | Split | LSTM Uni | LSTM Bi | Transformer | T5 | BART |
    """
    
    # Load the performance table
    if raw_table is not None:
        perf_table = raw_table
    else:
        perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')

    # Create row header
    rows = []
    row_header = ['Dataset', 'Split']
    for model in model_names:
        row_header.append(MODEL_NICE_NAMES[model])
    # rows.append(row_header)
    # Loop through the dataset and splits
    for dataset in dataset_and_splits:
        for split in dataset_and_splits[dataset]:
            curr_row = [DATASET_NICE_NAMES[dataset], SPLIT_NICE_NAMES[split]]
            for model in model_names:
                # Loop through the models, retrieve the corresponding performance
                if dataset == "COGS" and '-' in split:
                    data_split = split.split("-")[0]
                    eval_split = split.split("-")[1]
                    curr_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == data_split) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == eval_split) ]
                else:
                    curr_perf = perf_table.loc[(perf_table['Dataset'] == dataset) & (perf_table['Split'] == split) & (perf_table['Model'] == model)]
                if len(curr_perf) < 1:
                    # The training have not been finished yet
                    curr_row.append('IP')
                else:
                    if len(curr_perf) > 1:
                        # There should only be one result after the retrival
                        pdb.set_trace()
                    curr_row.append(round(curr_perf[metric_name].item() * 100.0, 1))
            rows.append(curr_row)
    return pd.DataFrame(rows, columns=row_header)

def plot_concurrence_distribution(coref='Kendall'):
    """
    Plot the concurrence value distribution, Legacy code, not used in paper
    """
    # Load the concurrence table
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    concurrences = helper_clean_concurrence_table(concurrences)
    
    # Plot the value distribution
    fig, ax = plt.subplots(figsize =(10, 7), tight_layout = True)
    _, _, bars = ax.hist(concurrences['concurrence'])
    ax.bar_label(bars)
    ax.grid(color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
    plt.title('Distribution of concurrence values; total pairs = ' + str(len(concurrences)))
    plt.xlabel('Concurrence value')
    plt.ylabel('Number of pairs')
    plt.savefig(os.environ['BASE_DIR'] + '/results/concur_dist.png')

def compute_corre_concur_perf(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    # Compute the correlation between concurrence and performane
    res = plot_perf_vs_concur(dataset_mapping=dataset_mapping, coref=coref)
    return np.corrcoef(res['perf'], res['concur'])[0, 1]

def plot_perf_vs_concur_split_type(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    """
    A scatter plot of performance on one dataset (averaged across models) / avg of the paired performance v.s. Concurrence - shape determined by the pairs type (Same Split vs Different split))
    Legacy code, not used anywhere
    """
    # Load performance and concurrence
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    # res will become a dataframe with cnocur | perf | type
    res = []
    for dataset1 in dataset_mapping:
        for dataset2 in dataset_mapping:
            eval_split1 = 'gen' if dataset1 == 'COGS' else 'test'
            eval_split2 = 'gen' if dataset2 == 'COGS' else 'test'
            # Loop through each split pair
            for split1 in dataset_mapping[dataset1]:
                for split2 in dataset_mapping[dataset2]:
                    # if dataset1 != dataset2 or split1 != split2:
                        # Compute average performance across models and datasets
                        setup1_perfs = perf_table.loc[(perf_table['Dataset'] == dataset1) & (perf_table['Split'] == split1) & (perf_table['Eval Split'] == eval_split1)]
                        setup2_perfs = perf_table.loc[(perf_table['Dataset'] == dataset2) & (perf_table['Split'] == split2) & (perf_table['Eval Split'] == eval_split2)]
                        current_concur = concurrences.loc[ (concurrences['Dataset1'] == dataset1) & (concurrences['Dataset2'] == dataset2) & (concurrences['Split1'] == split1) &(concurrences['Split2'] == split2) & (concurrences['EvalSplit1'] == eval_split1) & (concurrences['EvalSplit2'] == eval_split2)]
                        # Get pair types
                        if dataset1 != dataset2 and split1 == split2:
                            pair_type = 'DiffDsSameSplit'
                        elif dataset1 == dataset2 and split1 != split2:
                            pair_type = 'SameDsDiffSplit'
                        elif dataset1 != dataset2 and split1 != split2:
                            pair_type = 'DiffDsDiffSplit'
                        else:
                            pair_type = 'SameDsSameSplit'
                        if len(setup1_perfs) >= 1 and len(setup2_perfs) >= 1 and len(current_concur) == 1:
                            res.append({'concur': current_concur['concurrence'].item(),
                                        'perf': round((setup2_perfs['ignore_space'].mean() + setup1_perfs['ignore_space'].mean()) / 2 * 100, 2),
                                        'pair_type': pair_type
                                        })

    res = pd.DataFrame(res)
    # Start plotting
    type1_res = res.loc[res['pair_type'] == 'SameDsSameSplit']
    plt.scatter(type1_res['perf'].tolist(), type1_res['concur'].tolist(), c='#D81B60', alpha=0.5, label='Same Data and Split')
    type2_res = res.loc[res['pair_type'] == 'DiffDsSameSplit']
    plt.scatter(type2_res['perf'].tolist(), type2_res['concur'].tolist(), c='#6EA5F7', alpha=0.5, label='Different Data Same Split')
    type3_res = res.loc[res['pair_type'] == 'SameDsDiffSplit']
    plt.scatter(type3_res['perf'].tolist(), type3_res['concur'].tolist(), c='#FF9900', alpha=0.5, label='Same Data Different Split')
    type4_res = res.loc[res['pair_type'] == 'DiffDsDiffSplit']
    plt.scatter(type4_res['perf'].tolist(), type4_res['concur'].tolist(), c='#7570B3', alpha=0.5, label='Different Data and Split')

    plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.3)
    xticks = [x * 10 for x in range(11)]
    yticks = [x * 0.1 for x in range(-10, 11, 2)]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.legend()
    plt.xlabel('Average EM between two splits in the pair')
    plt.ylabel('Concurrence')
    plt.title(f'Performance-Concurrence Alignment')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/perf_alignment.png')
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/perf_alignment.pdf')

def plot_bar_concurrence_count_by_type(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    """
    Bar plot the count of concurrence by the type of pairs.
    Legacy
    """
    # Load Concurrence
    # Load performance and concurrence
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    # res will become a dataframe with cnocur | perf | type
    buckets = [0.1 * x for x in range(-10, 11, 2)]
    res = {
        'DiffDsSameSplit' : [0] * (len(buckets) - 1),
        'SameDsDiffSplit' : [0] * (len(buckets) - 1),
        'DiffDsDiffSplit' : [0] * (len(buckets) - 1),
        'SameDsSameSplit' : [0] * (len(buckets) - 1)
           }
    for dataset1 in dataset_mapping:
        for dataset2 in dataset_mapping:
            eval_split1 = 'gen' if dataset1 == 'COGS' else 'test'
            eval_split2 = 'gen' if dataset2 == 'COGS' else 'test'
            # Loop through each split pair
            for split1 in dataset_mapping[dataset1]:
                for split2 in dataset_mapping[dataset2]:
                    # if dataset1 != dataset2 or split1 != split2:
                        current_concur = concurrences.loc[ (concurrences['Dataset1'] == dataset1) & (concurrences['Dataset2'] == dataset2) & (concurrences['Split1'] == split1) &(concurrences['Split2'] == split2) & (concurrences['EvalSplit1'] == eval_split1) & (concurrences['EvalSplit2'] == eval_split2)]
                        # Get pair types
                        if dataset1 != dataset2 and split1 == split2:
                            pair_type = 'DiffDsSameSplit'
                        elif dataset1 == dataset2 and split1 != split2:
                            pair_type = 'SameDsDiffSplit'
                        elif dataset1 != dataset2 and split1 != split2:
                            pair_type = 'DiffDsDiffSplit'
                        else:
                            pair_type = 'SameDsSameSplit'
                        if len(current_concur) == 1:
                            current_concur = current_concur['concurrence'].item()
                            # Add the count to the dictonary
                            for idx, buc in enumerate(buckets[:-1]):
                                if current_concur >= buc and current_concur < buckets[idx + 1]:
                                    res[pair_type][idx] += 1
    # Get smallest and largest index that contains nonzero
    small_n_index = 20
    large_n_index = -1
    for pair_type in res:
        for idx in range(len(res[pair_type])):
            if res[pair_type][idx] != 0 and (idx == 0 or res[pair_type][idx - 1] == 0):
                small_n_index = idx if idx <= small_n_index else small_n_index
            if res[pair_type][idx] == 0:
                large_n_index = idx if idx >= large_n_index else large_n_index
    bucket_display = []
    for idx, buc in enumerate(buckets[:-1]):
        bucket_display.append('[' + str(round(buc, 1)) + ', ' + str(round(buckets[idx + 1], 1)) + ']')

    # Get the total amount of pairs, and normalize each instances
    for key in res:
        tot = sum(res[key])
        res[key] = [x / tot for x in res[key]]

    large_n_index += 1
    # Trim the data
    buckets = buckets[small_n_index: large_n_index]
    bucket_display = bucket_display[small_n_index: large_n_index]
    for pair_type in res:
        res[pair_type] = res[pair_type][small_n_index: large_n_index]
    helper_make_bar_plot(buckets, bucket_display, res)
    helper_make_normalized_nonstacked_bar_plot(buckets, bucket_display, res)

def compute_avg_per_dataset(dataset_mapping=default_dataset_mapping, coref='Kendall'):
    """
    Compute the average concurrence value for each dataset
    Return: A dict() with keys to be the dataset names, values to be the corresponding average
    """
    # Load concurrence
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    setup_to_show = helper_gen_data_split_str(dataset_mapping)
    concurrences = helper_clean_concurrence_table(concurrences, setup_to_show, exclude_self=True)
    # res will become a dataframe with cnocur | perf | type
    res = dict()
    for dataset in dataset_mapping:    
        # Loop through each split pair
        current_concur = concurrences.loc[(concurrences['Dataset1'] == dataset) | (concurrences['Dataset2'] == dataset)]
        res[dataset] = current_concur['concurrence'].mean(skipna = True)
    return res

def plot_performance_pretrainednonpretrained(dataset1, split1, dataset2, split2, save_plot=False):
    """
    Plot a figure like figure 1 in Liu et al, in which the x-axis is an dataset EM,
    and the y-axis is another dataset EM. Better to use concurrence values to determine
    pair to plot and show.
    Figure 5

    Will count each random seed trial as one variant
    """
    # Load the performance table
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/exact_match.csv')

    evalsplit1 = 'gen' if dataset1 == 'COGS' else 'test'
    evalsplit2 = 'gen' if dataset2 == 'COGS' else 'test'

    # For each model, record the corresponding performance in pair1 and pair2 as lists
    model_list = list(set(perf_table['Model'])) 
    pair1_perfs =[]
    pair1_non_pretrained = []
    pair1_pretrained = []
    pair2_perfs = []
    pair2_non_pretrained = []
    pair2_pretrained = []
    for model in model_list:
        # Filter the rows that satisfy the requriement
        pair1_perf = perf_table.loc[(perf_table['Dataset'] == dataset1) & (perf_table['Split'] == split1) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == evalsplit1) & (perf_table['Seed'] != 'Standard Deviation') & (perf_table['Seed'] != 'Mean')]
        pair2_perf = perf_table.loc[(perf_table['Dataset'] == dataset2) & (perf_table['Split'] == split2) & (perf_table['Model'] == model) & (perf_table['Eval Split'] == evalsplit2) & (perf_table['Seed'] != 'Standard Deviation') & (perf_table['Seed'] != 'Mean')]
        
        if len(pair1_perf['ignore_space']) == len(pair2_perf['ignore_space']) and 0.0 not in pair1_perf['ignore_space'] and 0.0 not in pair2_perf['ignore_space']:
            # When we have the nonzero performance of both pairs, record
            pair1_perfs += pair1_perf['ignore_space'].values.flatten().tolist()
            pair2_perfs += pair2_perf['ignore_space'].values.flatten().tolist()

            # Record to pretrained and nonpretrained
            if model in PRETRAINED_MODEL:
                pair1_pretrained += pair1_perf['ignore_space'].values.flatten().tolist()
                pair2_pretrained += pair2_perf['ignore_space'].values.flatten().tolist()
            elif model in UNPRETRAINED_MODEL:
                pair1_non_pretrained += pair1_perf['ignore_space'].values.flatten().tolist()
                pair2_non_pretrained += pair2_perf['ignore_space'].values.flatten().tolist()
    # Plotting
    if save_plot:
        # plt.scatter(pair1_perfs, pair2_perfs, c='#40B0A6', alpha=0.5)
        plt.scatter(pair1_pretrained, pair2_pretrained, c='#40B0A6', alpha=0.5, label='Pretrained Models')
        plt.scatter(pair1_non_pretrained, pair2_non_pretrained, c='#7570B3', alpha=0.5, label='Non-pretrained Models')
        plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4)
        # ticks = [x * 0.1 for x in range(11)]
        xticks = [x * 0.1 for x in range(int(max(max(pair1_pretrained), max(pair1_non_pretrained))) + 1)]
        yticks = [x * 0.1 for x in range(int(max(max(pair2_pretrained), max(pair2_non_pretrained)))) + 1]
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.legend()
        # TODO: Add legend of tau and gamma for the plotted dataset
        plt.xlabel('Performance on ' + dataset1 + ', split = ' + split1)
        plt.ylabel('Performance on ' + dataset2 + ', split = ' + split2)
        plt.title(f'{dataset1}-{split1} and {dataset2}-{split2}')
    
        plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/perf_alignment_{dataset1}_{split1}_{dataset2}_{split2}.png')
    return pair1_pretrained, pair2_pretrained, pair1_non_pretrained, pair2_non_pretrained


##### Beginning of Helper Functions
def helper_make_normalized_nonstacked_bar_plot(buckets, bucket_display, res):
    fig, ax = plt.subplots()
    width = 0.04
    bottom = np.zeros(len(buckets))
    colors = ["#96595A", "#DA897C", "#E4E4B2", "#B2E4CF", "#0D6A82"]
    plt.grid(color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.3)
    # def func(x, a, b, c):
    #     # a Gaussian distribution
    #     return a * np.exp(-(x-b)**2/(2*c**2))
    # Create the bar plot
    bucket_list = [buckets]
    for pair_type, weight_count in res.items():
        c = colors.pop()
        ax.bar(buckets, weight_count, width, label=pair_type, color=c)
        buckets = [x + width for x in buckets]
        # popt, pcov = curve_fit(func, buckets, weight_count)
        # x = np.linspace(-1, 1, 100)
        # y = func(x, *popt)

        # ax.plot(x + width/2, y, c=c)
    xticks = [bucket - 1.5*width for bucket in buckets]
    ax.set_xticks(xticks)
    # ax.set_xlim(xticks[0], xticks[-1])
    ax.set_xticklabels(bucket_display, rotation=30)
    ax.set_xlabel("Concurrence Range")
    ax.set_ylabel("Amount of Pairs over All Instances")
    # ax.set_title("Count of concurrence values")
    ax.legend()

    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/bar_count_non_stacked.png', bbox_inches = "tight")
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/bar_count_non_stacked.pdf', bbox_inches = "tight")

def helper_make_bar_plot(buckets, bucket_display, res):
    # Make the bar plot
    fig, ax = plt.subplots()
    width = 0.4
    bottom = np.zeros(len(buckets))
    colors = ["#96595A", "#DA897C", "#E4E4B2", "#B2E4CF", "#0D6A82"]
    for pair_type, weight_count in res.items():
        p = ax.bar(bucket_display, weight_count, width, label=pair_type, bottom=bottom, color=colors.pop())
        bottom += weight_count
    ax.set_xticklabels(bucket_display, rotation=30)
    ax.set_xlabel("Concurrence Range")
    ax.set_ylabel("Amount of Pairs")
    # ax.set_title("Count of concurrence values")
    ax.legend()

    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/bar_count.png', bbox_inches = "tight")
    plt.savefig(os.environ['BASE_DIR'] + f'/results/sanity/bar_count.pdf', bbox_inches = "tight")

def helper_clean_concurrence_table(concurrences, setup_to_show, exclude_self=True):
    new_concurrences = []
    processed_concurr = set()
    for _, concur_row in concurrences.iterrows():
        # Only include the concurrence in gen split
        if isinstance(concur_row['concurrence'], float) and (concur_row['Dataset1'] != 'COGS' or concur_row['EvalSplit1'] != 'test' or concur_row['Split1'] == 'length') and (concur_row['Dataset2'] != 'COGS' or concur_row['EvalSplit2'] != 'test' or concur_row['Split2'] == 'length'):
            
            # Record if this has not been recorded, and is not self-concurrence
            pair_str = concur_row['Dataset1'] + '-' + concur_row['Split1'] + '-' + concur_row['Dataset2'] + '-' + concur_row['Split2']
            if pair_str not in processed_concurr and concur_row["Dataset1"] + '-' + concur_row["Split1"] in setup_to_show and concur_row["Dataset2"] + '-' + concur_row["Split2"] in setup_to_show:
                if not exclude_self or (concur_row['Dataset1'] != concur_row['Dataset2'] or concur_row['Split1'] != concur_row['Split2']):
                    # Add to rows
                    new_concurrences.append(concur_row.values.flatten().tolist())

            # add reversed pair string because of symmetricity
            processed_concurr.add(concur_row['Dataset2'] + '-' + concur_row['Split2'] + '-' + concur_row['Dataset1'] + '-' + concur_row['Split1'])
            processed_concurr.add(pair_str)
    return pd.DataFrame(new_concurrences, columns=list(concurrences.columns))

def helper_gen_data_split_str(dataset_and_splits):
    """
    Help generating the dataset-split pair string in case some tables need different splits to be included
    """
    res = set()
    for dataset in dataset_and_splits:
        for split in dataset_and_splits[dataset]:
            if dataset != 'COGS':
                res.add(dataset + '-' + split)
            else:
                res.add(dataset + '-' + split.split('-')[0])
    return res

def helper_construct_colorbarticks(min_concur, max_concur):
    cticks = [-1.0]
    cticks_labels = [-1.0]
    if min_concur < 0.0:
        cticks.append(min_concur)
        cticks_labels.append('Min:\n ' + str(round(min_concur, 2)))
        if max_concur < 0.0:
            cticks.append(max_concur)
            cticks_labels.append('Max:\n ' + str(round(max_concur, 2)))
        cticks.append(0.0)
        cticks_labels.append(0.0)
    else:
        cticks.append(0.0)
        cticks_labels.append(0.0)
        cticks.append(min_concur)
        cticks_labels.append('Min:\n ' + str(round(min_concur, 2)))
    
    if max_concur > 0.0:
        cticks.append(max_concur)
        cticks_labels.append('Max:\n ' + str(round(max_concur, 2)))
    cticks.append(1.0)
    cticks_labels.append(1.0)
    return cticks, cticks_labels

def avg_concur(dataset_mapping, flag='avg_raw', coref='Kendall'):
    # Load the concurrence table
    if coref == 'Kendall':
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
    else:
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/concurrences.csv')
    setup_to_show = helper_gen_data_split_str(dataset_mapping)
    concurrences = helper_clean_concurrence_table(concurrences, setup_to_show, exclude_self=True)
    if flag == 'avg_mod':
        # Avg concur between modified splits
        current_concur1 = concurrences[concurrences['Split1'].str.contains('random_') & concurrences['Split2'].str.contains('random_')]
        current_concur1 = current_concur1[current_concur1["Dataset2"].isin(["COGS", "SCAN", "geoquery"])]
        current_concur1 = current_concur1[current_concur1["Dataset1"].isin(["COGS",  "SCAN","geoquery"])]
        # current_concur2 = concurrences.loc[('random_str' in concurrences['Split1'])]
        print(current_concur1['concurrence'].sum() / len(current_concur1))
    elif flag == 'avg_unmod':
        # Avg Concur between unmodified splits
        # Filter across splits
        current_concur1 = concurrences[concurrences["Split1"].isin(["no_mod", "addprim_turn_left", "standard", "tmcd"])]
        current_concur1 = current_concur1[current_concur1["Split2"].isin(["no_mod", "addprim_turn_left", "standard", "tmcd"])]
        current_concur1 = current_concur1[current_concur1["Dataset1"].isin(["COGS", "SCAN", "geoquery"])]
        current_concur1 = current_concur1[current_concur1["Dataset2"].isin(["COGS",  "SCAN","geoquery"])]
        print(current_concur1['concurrence'].sum() / len(current_concur1))
    else:
        # Compute everything except for those unedited
        concurrences = pd.read_csv(os.getenv('BASE_DIR') + '/results/Kendall_concurrences.csv')
        setup_to_show = helper_gen_data_split_str(raw_dataset_mapping)
        concurrences = helper_clean_concurrence_table(concurrences, setup_to_show, exclude_self=True)
        print(concurrences['concurrence'].sum() / len(concurrences))

def main():
    # Set plotting parameters
    font = {'family' : 'Times New Roman',
            # 'weight' : 'bold',
            'size'   : 11}
    mpl.rcParams['figure.dpi'] = 900
    mpl.rc('font', **font)
    mpl.rc('xtick', labelsize=13) 
    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rc('ytick', labelsize=13) 
    ### Legacy Genenerate Figures
    # avg_concur(dataset_mapping=all_dataset_mapping, flag='avg_d', coref='Kendall')
    # plot_perf_vs_concur_split_type(dataset_mapping=all_dataset_mapping)
    # plot_concurrence_distribution()
    # plot_bar_concurrence_count_by_type(dataset_mapping=all_dataset_mapping, coref='Kendall')
    # plot_performance_pretrainednonpretrained('SCAN', 'turn_left_random_cvcv', 'spider', 'tmcd')
    # plot_perf_synnat_pretrainednonpretrained(data_split_mapping=all_dataset_mapping)

    ### Legacy Generate Tables
    # Generate lexical dataset performance
    # res = gen_perf_plot(dataset_and_splits=lexical_dataset_mapping)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/lexical_perf_table_display.csv', index=False)
    # print(res)
    # Generate length split performance
    # res = gen_perf_plot(dataset_and_splits=length_dataset_mapping)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/length_perf_table_display.csv', index=False)
    # print(res)
    # res = gen_concurrence_table(dataset_and_splits=all_exclude_length_dataset_mapping, coref='Kendall', filter_type="top 5%")
    # print(res)
    # res = gen_perf_plot_w_std(dataset_and_splits=lexical_dataset_mapping)
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/lexical_perf_display.csv', index=False)
    
    #### Generate tables in paper
    # Figure 2
    # plot_nice_pubcount()
    # Figure 1 and 7: Plot confusion matrix of concurrences
    # Need to uncomment a few clause in the function for Fig 7
    # res = concurrence_conf_matrix(dataset_and_splits=default_dataset_mapping, coref='Kendall')
    # res = concurrence_conf_matrix(dataset_and_splits=all_dataset_mapping, coref='Kendall')
    # print(res)
    # Figure 4
    # plot_density_concurrence_count_by_type(dataset_mapping=all_dataset_mapping)
    # Figure 5
    # plot_lexical_perf_pretrainednonpretrained()
    # Figure 6
    # plot_perf_vs_concur()
    # Figure 3
    # plot_perf_synnat_pretrainednonpretrained(x_setups = ['SCAN-template_around_right', 'SCAN-addprim_turn_left', 'geoquery-template', 'SCAN-mcd1'], y_setups = ['spider-length', 'COGS-no_mod', 'spider-tmcd', 'spider-template'], save_name='')

    # Generate table and stats
    # Generate performance table with standard deviation - Table 1
    # res = gen_perf_plot_w_std()
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/perf_table_wStd_display.csv', index=False)
    # Table 12
    # res = gen_perf_plot_w_std(dataset_and_splits=all_dataset_mapping)
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/perf_table_wStd_all.csv', index=False)

    # Table 2: Generate the top 5% Concurrence values
    # res = gen_concurrence_table(dataset_and_splits=default_dataset_mapping, coref='Kendall', filter_type="0.7")
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/concurrence_display.csv', index=False)
    # Table 3
    # res = gen_concurrence_table(dataset_and_splits=length_dataset_mapping, coref='Kendall', filter_type="all")
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/concurrence_length_display.csv', index=False)
    # Table 4a
    # res = gen_concurrence_table(dataset_and_splits=lexical_dataset_mapping, coref='Kendall', filter_type="lexical split")
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/concurrence_lexical_display.csv', index=False)
    # Table 4b
    # res = gen_concurrence_table(dataset_and_splits=all_dataset_mapping, coref='Kendall', filter_type="0.7")
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/conurrence_lexical_high_display.csv', index=False)
    

    # Compute average to be displayed in Section Lexical
    # print(compute_corre_concur_perf(dataset_mapping=default_dataset_mapping))
    # print(compute_avg_per_dataset(dataset_mapping=default_dataset_mapping))
    # print(compute_avg_per_dataset(dataset_mapping=raw_dataset_mapping))
    # print(compute_avg_per_dataset(dataset_mapping=lexical_without_orig_dataset_mapping))

    ### Generate tables in appendix
    # res = get_spider_perf_appendix()
    # res.to_csv(os.getenv('BASE_DIR') + '/results/spider_raw_table.csv', index=False)
    # print(res)
    
    # res = gen_large_concurrence_table_appendix()
    # res.to_csv(os.getenv('BASE_DIR') + '/results/concurrence_full_display.csv', index=False)
    # print(res)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/concurrence_display.csv', index=False)

    # Generate performance table
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')
    res = get_eval_difference(perf_table, 'ignore_space', 'raw_exact_match')
    res = gen_eval_difference_display(res, all_dataset_mapping)
    print(res)
    res = gen_perf_plot(dataset_and_splits=all_dataset_mapping, model_names=default_model_names, metric_name='Diff', raw_table=res)
    res.to_csv(os.getenv('BASE_DIR') + '/results/perf_difference_display.csv', index=False)

    

if __name__ == "__main__":
    os.environ['BASE_DIR'] = '/private/home/kaisersun/CompGenComparision'
    main()
                
    