import pdb
import os
import re
import nltk
from tqdm import tqdm
import pandas as pd

from collections import Counter
from evaluate import load
import string
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from helper_utils.helper_methods import load_dataset_with_name, load_model_prediction, list_datasets_and_their_splits, load_processed_golds, get_model_list
# from analysis_utils import average_edit_distance

def helper_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction, ground_truth):
    prediction_tokens = helper_normalize_answer(prediction).split()
    ground_truth_tokens = helper_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def overall_f1(predictions, golds):
    f1 = 0.0
    for pred, gold in zip(tuple(predictions), tuple(golds)):
        f1 += f1_single(pred, gold)
    return f1/len(golds)

def evaluate_model(dataset_name, split, model_name, random_seed, eval_split='test', compute_edit_distance=False):
    """
    Evaluate a model based on the information given.
        eval_split = {'test', 'gen'}
        random seed" string
    Return a dict of EMs
    """
    exact_match = load("exact_match")
    # Load the gold and model prediction
    if 'lstm' in model_name or 'transformer' in model_name:
        golds = load_processed_golds(dataset_name, split, eval_split)
    else:
        dataset = load_dataset_with_name(dataset_name, split)
        golds = dataset[eval_split]['output']
    prediction = load_model_prediction(model_name, dataset_name, split, random_seed, eval_split)
    if eval_split == 'dev':
        eval_split = 'validation'
    ems = dict()
    golds = golds[:-1] if golds[-1] == '' else golds
    prediction = prediction[:-1] if prediction[-1] == '' else prediction
    # Format the gold and predion to feed into HF accuracy
    ems['raw_exact_match'] = exact_match.compute(predictions=prediction, references=golds)['exact_match']
    ems['ignore_case'] = exact_match.compute(predictions=prediction, references=golds, ignore_case=True)['exact_match']
    ems['ignore_case_and_punc'] = exact_match.compute(predictions=prediction, references=golds, ignore_case=True, ignore_punctuation=True)['exact_match']
    ems['ignore_space'] = exact_match.compute(predictions=prediction, references=golds, regexes_to_ignore=' ')['exact_match']
    ems['most_lenient'] = exact_match.compute(predictions=prediction, references=golds, ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')['exact_match']
    ems['ignore_right'] = exact_match.compute(predictions=prediction, references=golds, regexes_to_ignore=[' ', 'LEFT', 'RIGHT'])['exact_match']
    if compute_edit_distance:
        # Compute edit distance
        spacy = SpacyTokenizer()
        tok_prediction = [[t.text for t in spacy.tokenize(pred)] for pred in prediction]
        tok_golds = [[t.text for t in spacy.tokenize(gold)] for gold in golds]
        lev = 0.0
        print("Computing edit distance...")
        for idx, pred in tqdm(enumerate(tok_prediction)):
            lev += nltk.edit_distance(tok_golds[idx], pred) / len(tok_golds[idx])
        ems['avg_edit_distance'] = lev / len(tok_golds)
        # ems['avg_edit_distance'] = average_edit_distance(prediction, golds)
    ems['f1'] = overall_f1(prediction, golds)
    # For debugging, in order to check whether all random seeds are successfully computed
    ems['num_seed'] = -1
    return ems

def evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=False, compute_edit_distance=False):
    """
    Evaluate a model across all random seed
    """
    if 't5' in model_name or 'bart' in model_name or 'btg' in model_name:
        seeds = [0, 42, 12345]
    elif 'nqg' in model_name:
        seeds = [0, 42, 12345]
    elif 'transformer' in model_name:
        seeds = [0, 1, 42, 12345]
    else:
        seeds = [0, 1, 5, 42, 12345]
    basic_info = {
        'Model' : model_name,
        'Dataset' : dataset_name,
        'Split' : split,
        'Eval Split': eval_split
    }
    rows = []
    ems = None
    for seed in seeds:
        try:
            ems = evaluate_model(dataset_name, split, model_name, str(seed), eval_split, compute_edit_distance)
        except:
            print("\n\nThe model ", model_name, " has issue with seed", str(seed), ' on dataset ', dataset_name)
            if include_failed:
                ems = {
                    'raw_exact_match': -1,
                    'ignore_case': -1,
                    'ignore_case_and_punc': -1,
                    'ignore_space': -1,
                    'most_lenient': -1,
                    'f1':-1,
                    'num_seed':-1
                }
            else:
                continue
        ems['Seed'] = seed
        ems.update(basic_info)
        rows.append(ems)
    res = pd.DataFrame(rows)
    if ems:
        # Compute the mean and std of the ems items
        std = res.std(numeric_only=True).to_dict()
        mean = res.mean(numeric_only=True).to_dict()
        std['Seed'] = 'Standard Deviation'
        std['num_seed'] = len(res)
        mean['Seed'] = 'Mean'
        mean['num_seed'] = len(res)
        std.update(basic_info)
        mean.update(basic_info)
        res = res.append(std, ignore_index=True)
        res = res.append(mean, ignore_index=True)
    return res

def evaluate_model_across_datasets_and_splits(model_name, include_failed=False, compute_edit_distance=False):
    data_path = os.path.join(os.getenv("BASE_DIR"), 'data/')
    dataset_names, splits_mapping = list_datasets_and_their_splits(data_path)
    res = None
    for dataset_name in dataset_names:
        for split in splits_mapping[dataset_name]:
            if res is None:
                res = evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed, compute_edit_distance=compute_edit_distance)
            else:
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed)], ignore_index=True).drop_duplicates()
            if 'COGS' in dataset_name:
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='dev', include_failed=include_failed)], ignore_index=True).drop_duplicates()
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='gen', include_failed=include_failed)], ignore_index=True).drop_duplicates()
    return res

def evaluate_model_for_dataset(model_name, dataset_name, include_failed=False, compute_edit_distance=False):
    data_path = os.path.join(os.getenv("BASE_DIR"), 'data/')
    _, splits_mapping = list_datasets_and_their_splits(data_path)
    res = None
    for split in splits_mapping[dataset_name]:
        if res is None:
            res = evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed, compute_edit_distance=compute_edit_distance)
        else:
            res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
        if 'COGS' in dataset_name:
            res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='dev', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
            res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='gen', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
    return res

def evaluate_all_model_for_dataset(dataset_name, include_failed=False, compute_edit_distance=False):
    data_path = os.path.join(os.getenv("BASE_DIR"), 'data/')
    _, splits_mapping = list_datasets_and_their_splits(data_path)
    res = None
    model_names = get_model_list()
    for model_name in model_names:
        for split in splits_mapping[dataset_name]:
            if res is None:
                res = evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed, compute_edit_distance=compute_edit_distance)
            else:
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='test', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
            if 'COGS' in dataset_name:
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='dev', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
                res = pd.concat([res, evaluate_model_across_seeds(dataset_name, split, model_name, eval_split='gen', include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()
    return res

def evaluate_all(include_failed=False, compute_edit_distance=False):
    data_path = os.path.join(os.getenv("BASE_DIR"), 'data/')
    dataset_names, _ = list_datasets_and_their_splits(data_path)
    res = None
    for dataset_name in dataset_names:
        if res is None:
            res = evaluate_all_model_for_dataset(dataset_name, include_failed=include_failed, compute_edit_distance=compute_edit_distance)
        else:
            res = pd.concat([res, evaluate_all_model_for_dataset(dataset_name, include_failed=include_failed, compute_edit_distance=compute_edit_distance)], ignore_index=True).drop_duplicates()

    return res

def gen_performance_table(columns_to_keep, res_table):
    """
    Generate a presentable performance table, with only mean performance and standard deviation
        columns_to_keep: the list of column names in res_table to keep in this version
        res_table: The output of the evaluate funtions above (a pd Dataframe)
    """
    res = []
    for _, row in res_table.iterrows():
        if row['Seed'] == 'Mean':
            curr_row = {}
            curr_row['Dataset'] = row['Dataset']
            curr_row['Split'] = row['Split']
            curr_row['Model'] = row['Model']
            for column in columns_to_keep:
                curr_row[column] = row[column]
            curr_row['Eval Split'] = row['Eval Split']
            # Find corresponding standard deviation
            std = res_table.loc[(res_table['Seed'] == 'Standard Deviation') & (res_table['Dataset'] == row['Dataset']) & (res_table['Split'] == row['Split']) & (res_table['Model'] == row['Model']) & (res_table['Eval Split'] == row['Eval Split'])]
            if len(std) < 1:
                curr_row['Std'] = 'IP'
            else:
                curr_row['Std'] = str(round(std['ignore_space'].item(), 2))
            res.append(curr_row)
    return pd.DataFrame(res)

def main():
    # # Testing
    # ems = evaluate_model('COGS', 'no_mod', 'btg', '12345', eval_split='test', compute_edit_distance=False)
    # print(ems)
    # print(ems)
    # ems = evaluate_model_for_dataset('bart-base', 'SCAN', include_failed=False)
    # ems = evaluate_model_for_dataset('btg', 'COGS', include_failed=False)
    # ems = evaluate_model('SCAN', 'template_around_right', 'bart-base', '42', eval_split='test', compute_edit_distance=False)
    # print(ems)
    # ems.to_csv(os.getenv('BASE_DIR') + '/results/sanity/debug_bart.csv')
    # ems = evaluate_model_across_datasets_and_splits('bart-base', include_failed=False, compute_edit_distance=False)
    # ems.to_csv(os.getenv('BASE_DIR') + '/results/BART_table.csv')

    # ems = evaluate_model('COGS', 'no_mod', 'bart-base', random_seed='42', eval_split='test', compute_edit_distance=False)
    # print(ems)

    # res = evaluate_all(include_failed=True)
    # res.to_csv(os.getenv('BASE_DIR') + '/results/exact_match_include_failed.csv')

    res = evaluate_all(compute_edit_distance=False)
    res.to_csv(os.getenv('BASE_DIR') + '/results/exact_match.csv')

    res_table = pd.read_csv(os.getenv('BASE_DIR') + '/results/exact_match.csv')
    columns_to_keep = ['raw_exact_match', 'ignore_space', 'f1', 'ignore_right']
    res = gen_performance_table(columns_to_keep, res_table)
    res.to_csv(os.getenv('BASE_DIR') + '/results/perf_table.csv')

if __name__ == "__main__":
    os.environ['BASE_DIR'] = '/private/home/kaisersun/CompGenComparision'
    main()
