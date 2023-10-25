# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
import os
import re
import json
import pdb

from datasets import load_dataset

def load_dataset_with_name(dataset_name, split):
    """
    Take a dataset name and split name, load the dataset.
    Returns a huggingface dataset dict.
    """
    path = os.getenv("BASE_DIR") + '/data/' + dataset_name + '/' + split + '_split/'
    print(f"Loading {split} of {dataset_name} dataset")
    data_files = {}

    if os.path.exists(path + 'train.tsv'):
        data_files["train"] = path + 'train.tsv'
    if os.path.exists(path + 'dev.tsv'):
        data_files["validation"] = path + 'dev.tsv'
    if os.path.exists(path + 'test.tsv'):
        data_files["test"] = path + 'test.tsv'
    if os.path.exists(path + 'gen.tsv'):
        data_files["gen"] = path + 'gen.tsv'
    
    # Overwrite the file names if it's COGS dataset, which has 3 columns instead of 2
    if os.path.exists(path + 'nqg_train.tsv'):
        data_files["train"] = path + 'nqg_train.tsv'
    if os.path.exists(path + 'nqg_dev.tsv'):
        data_files["validation"] = path + 'nqg_dev.tsv'
    if os.path.exists(path + 'nqg_test.tsv'):
        data_files["test"] = path + 'nqg_test.tsv'
    if os.path.exists(path + 'nqg_gen.tsv'):
        data_files["gen"] = path + 'nqg_gen.tsv'

    # if 'COGS' in dataset_name:
    #     raw_datasets = load_dataset("csv", data_files=data_files, sep='\t', column_names=["input", "output", "Type"])
    # else:
    raw_datasets = load_dataset("csv", data_files=data_files, sep='\t', column_names=["input", "output"])
    return raw_datasets

def list_datasets_and_their_splits(data_path):
    """
    data_path (str): The directory that include all the dataset files
    returns:
        dataset_names (list of str)
        splits_mapping (dict, key in dataset_names): values are the available splits
    """
    avail_datasets = os.listdir(data_path)
    dataset_names = []
    splits_mapping = dict()

    for dir in avail_datasets:
        if 'orig' not in dir and '_hp' not in dir and 'processed_data' not in dir:
            dataset_names.append(dir)
            avail_splits = os.listdir(data_path +'/' + dir)
            # Add splits to the dict mapping
            for split in avail_splits:
                if '_split' in split:
                    if dir not in splits_mapping:
                        splits_mapping[dir] = []
                    splits_mapping[dir].append(re.sub('_split', '', split))
    return dataset_names, splits_mapping

def list_hardcode_datasets_and_their_splits():
    """
    In contrary to `list_datasets_and_their_splits`, this function list
    the datasets that are CURRENTLY in use.
    """
    dataset_names = ['COGS', 'geoquery', 'SCAN', 'spider', 'NACS']
    splits_mapping = {
        'COGS': ['no_mod', 'random_cvcv', 'random_str', 'length'],
        'geoquery': ['standard', 'standard_random_cvcv', 'standard_random_str', 'length',
                     'template', 'tmcd', 'tmcd_random_cvcv', 'tmcd_random_str'],
        # 'geoquery': ['standard', 'length', 'template', 'tmcd'],
        'SCAN': ['simple', 'length', 'mcd1', 'mcd2', 'mcd3', 'addprim_jump', 
                 'addprim_turn_left', 'jump_random_cvcv', 'jump_random_str',
                 'turn_left_random_cvcv', 'turn_left_random_str', 'template_around_right'],
        'spider': ['random', 'length', 'template', 'tmcd'],
        'NACS': ['simple', 'length', 'add_jump', 'add_turn_left']
    }
    return dataset_names, splits_mapping

def load_processed_golds(dataset_name, split, eval_split='test'):
    """
    To train with lstm and transformer from openNMT, need to process the data
    and the data order will be different from original.
    This function helps load the processed data.
    """
    path = os.getenv("BASE_DIR") + '/data/processed_data/' + dataset_name + '/' + split + '/'
    path += eval_split + '_target.txt'
    return open(path, 'r').read().split('\n')


def load_model_prediction(model_name, dataset_name, split, random_seed='0', eval_split='test'):
    if 'lstm' in model_name or 'transformer' in model_name:
        path = os.path.join(os.getenv("BASE_DIR"), 'preds/', dataset_name + '/', split + '/', eval_split + '_pred_1_example_' + model_name + '_s' + random_seed + '.txt')
    elif 't5' in model_name or 'bart' in model_name:
        path = os.path.join(os.getenv("BASE_DIR"), 'preds/', dataset_name + '/', split + '/', model_name + '_s' + str(random_seed) + '_' + eval_split + '.txt')
        # if eval_split == 'test' or eval_split == 'dev' and 'COGS' in dataset_name:
        #     path = os.path.join(os.getenv("BASE_DIR"), 'preds/', dataset_name + '/', split + '/', model_name + '_s' + str(random_seed) + '_' + eval_split + '.txt')
    elif 'nqg' in model_name:
        path = os.path.join(os.getenv("BASE_DIR"), 'preds/', dataset_name + '/', split + '/', 'nqg_' + eval_split + '_s' + random_seed + '.txt')
    elif 'btg' in model_name:
        path = os.path.join(os.getenv("BASE_DIR"), 'preds/', dataset_name + '/', split + '/', 'btg_' + random_seed + '_' + eval_split + '.txt')
    else:
        raise AttributeError('No such model supported: ' + model_name)
    
    predictions = open(path, 'r').read().split('\n')
    return predictions

def get_model_list():
    return ['t5-base', 'bart-base', 'lstm_bi', 'lstm_uni', 'transformer', 'btg']
