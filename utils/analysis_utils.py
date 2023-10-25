# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

import os
import json
import numpy as np
import pandas as pd
import pdb
import scipy.stats as stats

# from torchaudio.functional import edit_distance
from constants import DATA_DIR, MODEL_DIR
from helper_utils.helper_methods import list_datasets_and_their_splits, list_hardcode_datasets_and_their_splits, load_model_prediction, load_processed_golds

def load_training_curve_info(model_name, dataset, split, checkpoint=None, seed='0'):
    """
    Returns steps [list], ems [list], best_em float
    """
    ems = []
    steps = []
    best_em = 0.0

    # Find the path to the model
    path = os.path.join(MODEL_DIR, dataset, model_name + '_s' + seed + '_' + split + '_1e-4')
    
    if checkpoint is not None:
        path = os.path.join(path, 'checkpoint-' + checkpoint)
    # Load the model's trainer_state
    trainer_state = json.load(open(path + '/trainer_state.json'))
    for metrics in trainer_state['log_history']:
        if 'eval_exact_match' in metrics:
            ems.append(metrics['eval_exact_match'])
            steps.append(metrics['step'])
            if metrics['eval_exact_match'] > best_em:
                best_em = metrics['eval_exact_match']

    return steps, ems, best_em

def load_avg_training_curve_info(model_name, dataset, split, checkpoint=None):
    """
    Average the performance of training curve by random seeds
    Returns steps [list], ems [list], best_em float
    """
    res = None
    avg_best_em = 0.0
    old_step = -1
    for seed in ['0', '12345']:
        steps, ems, best_em = load_training_curve_info(model_name, dataset, split, checkpoint=checkpoint, seed=seed)
        if old_step != -1 and old_step != len(steps):
            raise Exception("The amount of training steps across random seeds are not identical.")
        if res is None:
            res = np.array(ems)
        old_step = len(steps)
        avg_best_em += best_em
    
    return steps, res / 3, avg_best_em / 3

def compute_concurrence(perf_table, dataset1, dataset2, split1, split2, corre_fn=None, eval_split1='test', eval_split2='test', metric_type='ignore_space', coref='Kendall'):
    """
    Input:
        perf_table: a pd dataframe that contains exact match, model name, splits, and seed
            should be an output from the functions in evaluation_utils.py
        dataset1, dataset2: strings that indicate the dataset name
        metric_type (str) = {'ignore_space', 'f1', 'raw_exact_match', 'ignore_case', 'ignore_case_and_punc', 'most_lenient'}
    """
    perf1 = {}
    perf2 = {}
    eval_split1 = eval_split1
    eval_split2 = eval_split2
    seeds = [0, 42, 12345, '0', '42', '12345']
    # Gather the performance of each model
    for _, row in perf_table.iterrows():
        if row['Dataset'] == dataset1 and row['Split'] == split1 and row['Seed'] != 'Mean' and row['Seed'] != 'Standard Deviation' and row['Eval Split'] == eval_split1 and row['Seed'] in seeds:
            if row['Model'] in perf1:
                perf1[row['Model']].append(row[metric_type])
            else:
                perf1[row['Model']] = [row[metric_type]]
        if row['Dataset'] == dataset2 and row['Split'] == split2 and row['Seed'] != 'Mean' and row['Seed'] != 'Standard Deviation' and row['Eval Split'] == eval_split2 and row['Seed'] in seeds:
            if row['Model'] in perf2:
                perf2[row['Model']].append(row[metric_type])
            else:
                perf2[row['Model']] = [row[metric_type]]
    
    # Linearize perf
    eval1 = []
    eval2 = []

    # In case there are some models haven't finish eval on all datasets, only 
    # compute on the available ones
    if len(perf1.keys()) >= len(perf2.keys()):
        model_dict = perf2
    else:
        model_dict = perf1
    
    for model in model_dict.keys():
        if model != 'nqg':
            # Not consider NQG for now
            # Loop through and recombine their performance
            for num1 in perf1[model]:
                for num2 in perf2[model]:
                    eval1.append(num1)
                    eval2.append(num2)
    if coref == 'Kendall':
        tau, p_value = stats.kendalltau(eval1, eval2)
        return tau
    return np.corrcoef(eval1, eval2)[0, 1]

def compute_concurr_all(metric_type='ignore_space', corref='Kendall'):
    dataset_names, splits_mapping = list_hardcode_datasets_and_their_splits()
    perf_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/exact_match.csv')
    rows = []

    setups = []
    for dataset in dataset_names:
        if dataset == 'COGS':
            eval_splits = ['test', 'gen']
        else:
            eval_splits = ['test']
        for split in splits_mapping[dataset]:
            for es in eval_splits:
                setups.append({
                    'dataset': dataset,
                    'split': split,
                    'es': es
                })
    
    for idx, setup1 in enumerate(setups):
            for setup2 in setups:
                curr_dic = {}
                try:
                    curr_dic['concurrence'] = compute_concurrence(perf_table, setup1['dataset'], setup2['dataset'], setup1['split'], setup2['split'], eval_split1=setup1['es'], eval_split2=setup2['es'], metric_type=metric_type, coref=corref)
                except:
                    continue
                curr_dic.update({
                                    'Dataset1': setup1['dataset'],
                                    'Split1': setup1['split'],
                                    'EvalSplit1': setup1['es'],
                                    'Dataset2': setup2['dataset'],
                                    'Split2': setup2['split'],
                                    'EvalSplit2': setup2['es']
                                    })
                rows.append(curr_dic)
    return pd.DataFrame(rows)

# def average_edit_distance(predictions, golds):
#     edit_dist = 0.0
#     for pred, gold in zip(predictions, golds):
#         if len(gold) != 0:
#             edit_dist += edit_distance(gold, pred) / len(gold)
#     return edit_dist / len(golds)

def prediction_agreement(model1, split1, eval_split1, model2, split2, eval_split2):
    """
    Compute the prediction agreement between two models
    """
    pass

def get_eval_difference(perf_table, metric1, metric2):
    """
    Generate a dataframe of Dataset Split Model Metric_difference
    metric1, 2: str of metric names
    perf_table: The processed, averaged performance table with Dataset,Split,Model,ignore_space,f1,ignore_right,Eval Split,Std
    """
    res_rows = []
    for _, row in perf_table.iterrows():
        res_rows.append({
            'Dataset': row['Dataset'],
            'Split': row['Split'],
            'Model': row['Model'],
            'Eval Split': row['Eval Split'],
            'Diff': row[metric1] - row[metric2]
        })
    return pd.DataFrame(res_rows)

def find_BTG_error_instance(split):
    # Load the simple split prediction
    simple_prediction = load_model_prediction('btg', 'SCAN', split, '0', 'test')
    simple_golds = load_processed_golds('SCAN', split, 'test')
    simple_golds = simple_golds[:-1] if simple_golds[-1] == '' else simple_golds
    simple_prediction = simple_prediction[:-1] if simple_prediction[-1] == '' else simple_prediction
    for pred, gold in zip(simple_prediction, simple_golds):
        if pred.replace(" ", "").replace("LEFT", "").replace("RIGHT", "") == gold.replace(" ", "").replace("LEFT", "").replace("RIGHT", ""):
            print(pred)
            print(gold)
    
if __name__ == "__main__":
    find_BTG_error_instance(split='addprim_turn_left')
