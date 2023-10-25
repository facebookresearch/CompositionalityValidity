# This file include utility functions to compute stats given a dataset
import re
import os
import csv
import pdb
import nltk
import pandas as pd

from tqdm import tqdm
from prettytable import PrettyTable
from torchaudio.functional import edit_distance
from transformers import AutoTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from helper_utils.helper_methods import load_dataset_with_name, list_hardcode_datasets_and_their_splits, list_datasets_and_their_splits
from constants import DATA_DIR

def build_table_for_all_datasets(data_type, sub_datatype=None, model_name='t5-base'):
    """
    Build a csv table for all data & splits
    Input:
        Data_type in {num_instances, raw_avg_length, tok_seq_length, lexical_overlap}
    """
    base_dir = os.getenv('BASE_DIR')

    # Construct table with dataset stats
    tab = PrettyTable(header=True)
    optim_splits = ['train', 'validation', 'test', 'gen', 'Overall']
    tab.field_names = ['Dataset', 'Split'] + optim_splits

    dataset_names, splits_mapping = list_hardcode_datasets_and_their_splits()

    for dataset_name in dataset_names:
        for split in splits_mapping[dataset_name]:
            curr_row = []
            curr_row.append(dataset_name)
            curr_row.append(split)
            # Compute the data
            if data_type == 'num_instances':
                res, _ = number_of_instances(dataset_name, split)
            elif data_type == 'raw_avg_length':
                input_avg_len, output_avg_len, _ = compute_avg_length(dataset_name, split)
                if sub_datatype == 'input':
                    res = input_avg_len
                else:
                    res = output_avg_len
            elif data_type == 'tok_seq_length':
                input_avg_len, output_avg_len, _ = compute_avg_tokenized_length_hf(dataset_name, split, target_model_name = model_name)
                if sub_datatype == 'input':
                    res = input_avg_len
                else:
                    res = output_avg_len
            elif data_type == 'lexical_overlap':
                res, _ = compute_lexical_overlap(dataset_name, split, lex_type='Jaccard')
            else:
                raise ValueError('The data_type can only be {num_instances, raw_avg_length, tok_seq_length, lexical_overlap}.')
            
            # Build the table
            for optim_split in optim_splits:
                if optim_split in res:
                    curr_row.append(res[optim_split])
                elif optim_split == 'Overall' and 'avg' in res:
                    # For seq length, the overall equiv to avg
                    curr_row.append(res['avg'])
                else:
                    curr_row.append('-')
            tab.add_row(curr_row)
    if not os.path.exists(base_dir + '/results/analysis_res/'):
        os.makedirs(base_dir + '/results/analysis_res/')

    # Construct CSV filename
    file_name = data_type
    if sub_datatype:
        file_name += '_' + sub_datatype
    if data_type == 'tok_seq_length':
        if '/' in model_name:
            file_name += '_' + model_name.split('/')[-1]
        else:
            file_name += '_' + model_name

    with open(base_dir + '/results/analysis_res/' + file_name + '.csv', 'w', newline='') as f:
        f.write(tab.get_csv_string())
    print(tab)

def number_of_instances():
    """
    Output number of instances for each dataset

    Outputs:
        avg_overlap (dict): avg lexical overlap between input and output, keys are train/test/dev
    """
    base_dir = os.getenv('BASE_DIR')

    # Construct table with dataset stats
    tab = PrettyTable()
    optim_splits = ['train', 'validation', 'test', 'gen', 'Overall']
    tab.add_row(['Dataset', 'Split'] + optim_splits)

    dataset_names, splits_mapping = list_hardcode_datasets_and_their_splits()

    for dataset_name in dataset_names:
        for split in splits_mapping[dataset_name]:
            curr_row = []
            # Load the dataset
            dataset = load_dataset_with_name(dataset_name, split)
            curr_row.append(dataset_name)
            curr_row.append(split)
            for optim_split in optim_splits:
                if optim_split in dataset:
                    curr_row.append(len(dataset[optim_split]))
                else:
                    curr_row.append(0.0)
            # Add up the instance count for overal
            curr_row[-1] = sum(curr_row[2:])
            tab.add_row(curr_row)
    
    if not os.path.exists(base_dir + '/results/analysis_res/'):
        os.makedirs(base_dir + '/results/analysis_res/')
    with open(base_dir + '/results/analysis_res/num_instances.csv', 'w', newline='') as f:
        f.write(tab.get_csv_string())    
    print(tab)

def number_of_instances(dataset_name, split):
    """
    Output number of instances for each dataset

    Outputs:
        num_instances (dict): number of instance in each optimization split, keys are train/test/dev
    """
    # Construct table with dataset stats
    tab = PrettyTable()
    num_instances = dict()
    split_names = []
    overall_num = 0
    num_column = []

    # Load the dataset
    dataset = load_dataset_with_name(dataset_name, split)

    for optim_split in dataset:
        split_names.append(optim_split)
        num_instances[optim_split] = len(dataset[optim_split])
        num_column.append(len(dataset[optim_split]))
        overall_num += len(dataset[optim_split])
    # Add the instance count for overal
    num_column.append(overall_num)
    num_instances['Overall'] = overall_num
    tab.add_column('Split', split_names + ['Overall'])
    tab.add_column('Number of Instances',  num_column)
    
    return num_instances, tab

def compute_avg_length(dataset_name, split):
    """
    Computes the average number of words of input and output

    Outputs:
        input_avg_len (dict): avg number of words in input, keys are train/test/dev
        output_avg_len (dict): avg number of words in output, keys are train/test/dev
        tab (PrettyTable): the table with a display of dataset stat 
    """

    # TODO: Maybe plot the distribution of length, too?
    # Load the dataset
    dataset = load_dataset_with_name(dataset_name, split)

    # Construct table with dataset stats
    tab = PrettyTable()
    input_avg_len = dict()
    output_avg_len = dict()

    # Loop through the split
    split_names = []
    dataset_lens = []
    input_lens_column = []
    output_lens_column = []
    overall_input_len = 0
    overall_output_len = 0

    for ft_split in dataset:
        split_names.append(ft_split)
        dataset_lens.append(len(dataset[ft_split]))
        tot_input_len = 0
        tot_output_len = 0
        for instance in dataset[ft_split]:
            tot_input_len += len(re.findall(r'\w+', instance['input']))
            tot_output_len += len(re.findall(r'\w+', instance['output']))
        
        input_avg_len[ft_split] = tot_input_len / len(dataset[ft_split])
        output_avg_len[ft_split] = tot_output_len / len(dataset[ft_split])
        input_lens_column.append(input_avg_len[ft_split])
        output_lens_column.append(output_avg_len[ft_split])
        overall_input_len += tot_input_len
        overall_output_len += tot_output_len
    # Add the averaged length to table data for display
    input_lens_column.append(overall_input_len / sum(dataset_lens))
    output_lens_column.append(overall_output_len / sum(dataset_lens))
    input_avg_len['avg'] = input_lens_column[-1]
    output_avg_len['avg'] = output_lens_column[-1]

    tab.add_column('Split', split_names + ['Overall'])
    tab.add_column('Number of Instances', dataset_lens + [0])
    tab.add_column('Avg input length', input_lens_column)
    tab.add_column('Avg output length', output_lens_column)
 
    return input_avg_len, output_avg_len, tab

def compute_lexical_overlap(dataset_name, split, lex_type="Lev"):
    """
    Computes the average lexical overlap (Levenshtein distance / input_len) between input and output
    lex_type = {Lev, Jaccard}
    Outputs:
        avg_overlap (dict): avg lexical overlap between input and output, keys are train/test/dev
    """
    # Load the dataset
    dataset = load_dataset_with_name(dataset_name, split)
    spacy = SpacyTokenizer()

    # Construct table with dataset stats
    tab = PrettyTable()
    avg_overlap = dict()

    # Loop through the split
    split_names = []
    dataset_lens = []
    overlap_column = []
    overall_overlap = 0.0

    for ft_split in dataset:
        split_names.append(ft_split)
        dataset_lens.append(len(dataset[ft_split]))
        tot_overlap = 0.0
        for instance in tqdm(dataset[ft_split]):
            if lex_type == 'Lev':
                tot_overlap += edit_distance(instance['input'], instance['output']) / len(instance['input'])
            else:
                tokenized_original = [t.text for t in spacy.tokenize(instance['input'])]
                tokenized_edited = [t.text for t in spacy.tokenize(instance['output'])]
                lev = nltk.edit_distance(tokenized_original, tokenized_edited)
                tot_overlap += lev/len(tokenized_original)
        avg_overlap[ft_split] = tot_overlap / len(dataset[ft_split])
        overlap_column.append(avg_overlap[ft_split])
        overall_overlap += tot_overlap
    # Add the averaged length to table data for display
    overlap_column.append(overall_overlap / sum(dataset_lens))
    avg_overlap['avg'] = overlap_column[-1]

    tab.add_column('Split', split_names + ['Overall'])
    tab.add_column('Number of Instaces', dataset_lens + [0])
    tab.add_column('Avg Lev(input, output) / input_len', overlap_column)

    return avg_overlap, tab

def sample_and_compute_overlap(dataset_name1, split1, dataset_name2, split2, lex_type="Lev"):
    """
    Samples some instances from these two datasets and compute the overlap between these instances
    lex_type = {Lev, Jaccard}
    Outputs:
        avg_overlap (dict): avg lexical overlap between the randomly sampled instances, keys are train/test/dev
    """
    # Load the dataset
    dataset1 = load_dataset_with_name(dataset_name1, split1)
    dataset2 = load_dataset_with_name(dataset_name2, split2)
    spacy = SpacyTokenizer()

    # Construct table with dataset stats
    tab = PrettyTable()
    avg_overlap = dict()
    boundary = 200

    # Sample the instances from eval_sets
    # Shuffle
    if 'gen' in dataset1:
        eval_split1 = 'gen'
    else:
        eval_split1 = 'test'
    if 'gen' in dataset2:
        eval_split2 = 'gen'
    else:
        eval_split2 = 'test'

    dataset1[eval_split1].shuffle(seed=0)
    dataset2[eval_split2].shuffle(seed=0)

    # Loop through the split
    tot_in_overlap = 0.0
    tot_out_overlap = 0.0
    for idx, instance1 in tqdm(enumerate(dataset1[eval_split1])):
        if idx < boundary:
            instance2 = dataset2[eval_split2][idx]
            if lex_type == 'Lev':
                tot_in_overlap += edit_distance(instance1['input'], instance2['input']) / max(len(instance1['input']), len(instance2['input']))
                tot_out_overlap += edit_distance(instance1['output'], instance2['output']) / max(len(instance1['output']), len(instance2['output']))
            else:
                tokenized_original = [t.text for t in spacy.tokenize(instance1['input'])]
                tokenized_edited = [t.text for t in spacy.tokenize(instance2['input'])]
                lev = nltk.edit_distance(tokenized_original, tokenized_edited)
                tot_in_overlap += lev/max(len(tokenized_original), len(tokenized_edited))

                tokenized_original = [t.text for t in spacy.tokenize(instance1['output'])]
                tokenized_edited = [t.text for t in spacy.tokenize(instance2['output'])]
                lev = nltk.edit_distance(tokenized_original, tokenized_edited)
                tot_out_overlap += lev/max(len(tokenized_original), len(tokenized_edited))
    tot_in_overlap = tot_in_overlap / boundary
    tot_out_overlap = tot_out_overlap / boundary

    tab.add_column('Dataset1', [dataset_name1])
    tab.add_column('Split1', [split1])
    tab.add_column('Dataset2', [dataset_name2])
    tab.add_column('Split2', [split2])
    tab.add_column('Input Overlap', [tot_in_overlap])
    tab.add_column('Output Overlap', [tot_out_overlap])

    ret_dict = {
        'Dataset1': dataset_name1,
        'Split1': split1,
        'Dataset2': dataset_name2,
        'Split2': split2,
        'Input Overlap': tot_in_overlap,
        'Output Overlap': tot_out_overlap
    }
    return ret_dict, tab

def compute_sample_overlap_all_datasets(lex_type="Lev"):
    dataset_names, splits_mapping = list_hardcode_datasets_and_their_splits()
    setups = []
    for dataset in dataset_names:
        for split in splits_mapping[dataset]:
            setups.append({
                'dataset': dataset,
                'split': split
            })
    rows = []
    for idx, setup1 in enumerate(setups[:-1]):
        for setup2 in setups[idx + 1:]:
            ret_dict, _ = sample_and_compute_overlap(setup1['dataset'], 
                                                       setup1['split'], 
                                                       setup2['dataset'], 
                                                       setup2['split'], 
                                                       lex_type=lex_type)
            rows.append(ret_dict)
    return pd.DataFrame(rows)
    
def compute_avg_tokenized_length_hf(dataset_name, split, target_model_name, max_seq_length=512, max_output_length=512):
    """
    Computes the average number of tokens of input and output after tokenization
    Inputs:
        dataset_name={COGS, geoquery, spider, SCAN}
        target_model_name=model name from Huggingface that has a tokenizer or a path

    Outputs:
        input_avg_len (dict): avg number of tokens in input, keys are train/test/dev
        output_avg_len (dict): avg number of tokens in output, keys are train/test/dev
    """
    # Construct table with dataset stats
    tab = PrettyTable()
    input_avg_len = dict()
    output_avg_len = dict()

    # Load the dataset
    dataset = load_dataset_with_name(dataset_name, split)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, use_fast=True)

    # Loop through the split
    split_names = []
    dataset_lens = []
    input_lens_column = []
    output_lens_column = []
    overall_input_len = 0
    overall_output_len = 0

    for optim_split in dataset:
        # Tokenize
        inputs = dataset[optim_split]['input']
        if 't5' in target_model_name:
            inputs = ['semanticparse: ' + x for x in inputs]
        else:
            inputs = [x for x in inputs]
        model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(dataset[optim_split]['output'], max_length=max_output_length, truncation=True)
        # Compute the length
        split_names.append(optim_split)
        dataset_lens.append(len(dataset[optim_split]))
        tot_input_len = 0
        tot_output_len = 0
        for input_tok, output_tok in zip(model_inputs['input_ids'], labels['input_ids']):
            tot_input_len += len(input_tok)
            tot_output_len += len(input_tok)
        
        input_avg_len[optim_split] = tot_input_len / len(dataset[optim_split])
        output_avg_len[optim_split] = tot_output_len / len(dataset[optim_split])
        input_lens_column.append(input_avg_len[optim_split])
        output_lens_column.append(output_avg_len[optim_split])
        overall_input_len += tot_input_len
        overall_output_len += tot_output_len
    # Add the averaged length to table data for display
    input_lens_column.append(overall_input_len / sum(dataset_lens))
    output_lens_column.append(overall_output_len / sum(dataset_lens))
    input_avg_len['avg'] = input_lens_column[-1]
    output_avg_len['avg'] = output_lens_column[-1]

    tab.add_column('Split', split_names + ['Overall'])
    tab.add_column('Number of Instances', dataset_lens + [0])
    tab.add_column('Avg input length', input_lens_column)
    tab.add_column('Avg output length', output_lens_column)

    return input_avg_len, output_avg_len, tab

def main():
    build_table_for_all_datasets("num_instances")
    # res = compute_sample_overlap_all_datasets(lex_type="Lev")
    # res.to_csv(os.getenv('BASE_DIR') + '/results/dataset_overlap_Lev.csv')

    # res = compute_sample_overlap_all_datasets(lex_type="SpacyLev")
    # res.to_csv(os.getenv('BASE_DIR') + '/results/dataset_overlap_SpacyLev.csv')

if __name__ == "__main__":
    os.environ['BASE_DIR'] = '/private/home/kaisersun/CompGenComparision'
    main()
