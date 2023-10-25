"""Reformats .tsv format into .json that is compatible with the parsing-transformers codebase"""
"""Source: https://github.com/najoungkim/cogs-with-pretraining/blob/main/utils/convert_cogs_to_json.py"""
import argparse
import json
import os

_SPLITS = ['test', 'dev', 'train', 'gen', 'train_100', 'train_exposure_examples', 'test_heldout_items']

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default=None, type=str, required=True,
                        help='Path to directory containing the COGS dataset.')
    parser.add_argument('--output_path', default=None, type=str, required=True,
                        help='Path to save the output data to.')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    for split in _SPLITS:
        lines_to_write = []
        with open(os.path.join(args.input_path, f'{split}.tsv')) as f:
            for line in f:
                sent, lf, gen_type = line.rstrip('\n').split('\t')
                json_str = json.dumps({'translation': 
                                        {'en': sent,
                                         'mentalese': lf}
                                      })
                lines_to_write.append(json_str + '\n')

        with open(os.path.join(args.output_path, f'{split}.json'), 'w') as wf:
            wf.writelines(lines_to_write)


if __name__ == '__main__':
    main()