# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

"""Reformats COGS dataset with different variations of held-out lexical items."""

import argparse
import os
import string
import random

# Prespecified set of file names in the COGS dataset.
_DATASET_FILENAMES = ['train', 'test', 'dev', 'gen', 'train_100', 'test_heldout_items']
_HELD_OUT_VOCAB_MAP = {
    'hedgehog': '[w_0]',
    'shark': '[w_1]',
    'cockroach': '[w_2]',
    'Lina': '[w_3]',
    'Paula': '[w_4]',
    'Charlie': '[w_5]',
    'baked': '[w_6]',
    'bake': '[w_7]',
    'shattered': '[w_8]',
    'shatter': '[w_9]',
    'crawl': '[w_10]',
    'cobra': '[w_11]',
    'hippo': '[w_12]',
    'blessed': '[w_13]',
    'bless': '[w_14]',
    'squeezed': '[w_15]',
    'squeeze': '[w_16]',
    'teleported': '[w_17]',
    'teleport': '[w_18]',
    'shipped': '[w_19]',
    'ship': '[w_20]'
}

_DATASET_FILENAMES_NON_COGS = ['train', 'test']
_HELD_OUT_VOCAB_MAP_SCAN_JUMP = {
    'jump': '[w_0]',
    'I_JUMP': '[w_1]'
}

_HELD_OUT_VOCAB_MAP_SCAN_TURN_LEFT = {
    'left': '[w_0]',
    'I_TURN_LEFT': '[w_1]'
}

_RANDOM_STR_LENS = range(15, 30)
_RANDOM_STR_LENS_SHORTER = range(7, 15)
_LONG_WID_LENS = range(150, 160)
_CONSONANTS = 'bcdfghjklmnpqrstvwxyz'
_CONSONANTS_SET2 = 'bcdfghklmnprstvwz'
_VOWELS = 'aeiou'

def generate_held_out_vocab_map(words):
    vocab_map = {}
    for idx, word in enumerate(words):
        vocab_map[word] = '[w_' + str(idx) + ']'
    return vocab_map

#  python reformat_lexical_heldouts.py --dataset 'SCAN' --split 'jump' --input_path 'data/SCAN/addprim_jump_split/' --output_path 'data/SCAN/jump_random_str/' --new_heldout_type 'random_str'
# python reformat_lexical_heldouts.py --dataset 'geoquery' --split 'standard' --input_path 'data/geoquery/standard_split/' --output_path 'data/geoquery/standard_random_cvcv/' --new_heldout_type 'random_cvcv_str' --held_out_vocab 'data/geoquery/atoms.txt'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='COGS', type=str, required=True,
                        help='Dataset names', choices=['COGS', 'SCAN', 'geoquery', 'spider'])
    parser.add_argument('--split', default='standard', type=str, required=True,
                        help='The split to convert', choices=['standard', 'jump', 'turn_left', 'standard', 'tmcd'])

    parser.add_argument('--input_path', default=None, type=str, required=True,
                        help='Path to directory containing the COGS dataset.')
    parser.add_argument('--output_path', default=None, type=str, required=True,
                        help='Path to save the output data to.')
    parser.add_argument('--new_heldout_type', default='[w_n]', type=str,
                        choices=['no_mod', '[w_n]', '[w_n]_randn', 'random_str', 'random_str_shorter',
                                 'random_cvcv_str', 'random_cvcv_str_shorter',
                                 'random_cvcv_str_cons_reduced', 'random_cvcv_str_cons_reduced_shorter'],
                        help='Type of replacement for the held-out lexical items.')
    parser.add_argument('--initial_extra_space', default='no_space', type=str,
                        choices=['no_space', 'only_initial_novel_word', 'all_initial_words'],
                        help='Whether to add sentence-initial spacing (for tokenizer consistency).')
    parser.add_argument('--oversample_exposure_examples', action='store_true',
                        help='If true, oversample exposure examples in the training set.')
    parser.add_argument('--seed', default=555, type=int,
                        help='Random seed for random character sampling.')
    parser.add_argument('--held_out_vocab', default=None, type=str,
                        help='The path to held out vocab txt file. This happens for GEOQUERY and SPIDER since they have too much atoms.')
    args = parser.parse_args()

    if args.dataset == 'COGS':
        held_out_vocab_map = _HELD_OUT_VOCAB_MAP
        ds_filenames = _DATASET_FILENAMES
    elif args.dataset == 'SCAN':
        # SCAN
        ds_filenames = _DATASET_FILENAMES_NON_COGS
        if args.split == 'jump':
            held_out_vocab_map = _HELD_OUT_VOCAB_MAP_SCAN_JUMP
        else:
            # turn left
            held_out_vocab_map = _HELD_OUT_VOCAB_MAP_SCAN_TURN_LEFT
    else:
        ds_filenames = _DATASET_FILENAMES_NON_COGS

        vocabs = open(args.held_out_vocab, "r").read().split('\n')
        held_out_vocab_map = generate_held_out_vocab_map(vocabs)
        

    # Create a new directory if the specified output path does not exist.
    os.makedirs(args.output_path, exist_ok=True)

    random.seed(args.seed)
    vocab_map = None

    if args.new_heldout_type == 'no_mod':
        vocab_map = {key: key for key in held_out_vocab_map}
    elif args.new_heldout_type == '[w_n]':
        vocab_map = held_out_vocab_map
    elif args.new_heldout_type == '[w_n]_randn':
        vocab_map = dict.fromkeys(held_out_vocab_map.keys())
        w_vals = {}
        for key in vocab_map.keys():
            word_idx = ''
            word_idx_len = random.choice(_LONG_WID_LENS)
            while word_idx in vocab_map.values() or word_idx == '':
                for _ in range(word_idx_len):
                    word_idx += str(random.randint(0, 9))
                vocab_map[key] = f'[w_{word_idx}]'
        assert len(vocab_map.values()) == len(set(vocab_map.values())), (len(vocab_map.values()), len(set(vocab_map.values())))
    elif args.new_heldout_type.startswith('random_str'):
        vocab_map = {}
        for key in held_out_vocab_map.keys():
            if args.new_heldout_type == 'random_str':
                word_len = random.choice(_RANDOM_STR_LENS) 
            else:
                word_len = random.choice(_RANDOM_STR_LENS_SHORTER)
            new_word = ''
            for _ in range(word_len):
                new_word += random.choice(string.ascii_letters).lower()
            vocab_map[key] = new_word
    elif args.new_heldout_type.startswith('random_cvcv_str'):
        vocab_map = {}
        for key in held_out_vocab_map.keys():
            if args.new_heldout_type.endswith('shorter'):
                word_len = random.choice(_RANDOM_STR_LENS_SHORTER)
            else:
                word_len = random.choice(_RANDOM_STR_LENS)
            if 'cons_reduced' in args.new_heldout_type:
                consonant_set = _CONSONANTS_SET2
            else:
                consonant_set = _CONSONANTS
            new_word = ''
            for i in range(word_len):
                if i % 2 == 0:
                    new_word += random.choice(consonant_set)
                else:
                    new_word += random.choice(_VOWELS)
            vocab_map[key] = new_word
    else:
        raise NotImplementedError

    # Map the words to the newly crafted fake words
    print(vocab_map)
    for filename in ds_filenames:
        lines_to_write = []
        exposure_examples = []
        with open(os.path.join(args.input_path, f'{filename}.tsv')) as f:
            for line in f:
                write_exposure_example = False
                if args.dataset == 'COGS':
                    source, target, gen_type = line.rstrip('\n').split('\t')
                else:
                    source, target = line.rstrip('\n').split('\t')
                if any([vocab_map.get(w, False) for w in source.split()]):
                    write_exposure_example = True
                source = ' '.join([vocab_map.get(w, w) for w in source.split()])
                target = ' '.join([vocab_map.get(w, w) for w in target.split()])
                if args.new_heldout_type in ['[w_n]', '[w_n]_randn', 'random_cvcv_str_shorter']:
                    if args.initial_extra_space == 'all_initial_words':
                        source = ' ' + source
                        target = ' ' + target
                    elif args.initial_extra_space == 'only_initial_novel_word':
                        if source.startswith(tuple(vocab_map.values())):
                            source = ' ' + source
                        if target.startswith(tuple(vocab_map.values())):
                            target = ' ' + target
                # Remove the gen_type
                if args.dataset == 'COGS':
                    lines_to_write.append(f'{source}\t{target}\t{gen_type}\n')
                    if 'train' in filename and write_exposure_example:
                        exposure_examples.append(f'{source}\t{target}\t{gen_type}\n')
                        if args.oversample_exposure_examples:
                            for _ in range(0, 200):
                                lines_to_write.append(f'{source}\t{target}\t{gen_type}\n')
                else:
                    lines_to_write.append(f'{source}\t{target}\n')
                    if 'train' in filename and write_exposure_example:
                        exposure_examples.append(f'{source}\t{target}\n')
                        if args.oversample_exposure_examples:
                            for _ in range(0, 200):
                                lines_to_write.append(f'{source}\t{target}\n')

        with open(os.path.join(args.output_path, f'{filename}.tsv'), 'w') as wf:
            wf.writelines(lines_to_write)

        if 'train' in filename:
            with open(os.path.join(args.output_path, f'{filename}_exposure_examples.tsv'), 'w') as wf:
                wf.writelines(exposure_examples)


    print(f'Reformatted and saved COGS data for T5 to {args.output_path}.')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, 'added_vocab.txt'), 'w') as wf:
        for vocab in vocab_map.values():
            wf.write(f'{vocab}\n')


if __name__ == '__main__':
    main()
