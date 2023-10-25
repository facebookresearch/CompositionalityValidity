# Remove redudant checkpoints
import pdb
import os
import shutil
import json
from glob import glob
from tqdm import tqdm

def remove_redudant_checkpoints_hf(model_path):
    """
    Remove redundant checkpoints in the given model_path, keep only the best and the last checkpoint
    """
    # Load trainer_state
    dir_list = glob(model_path + "/*/")
    # Find path to best checkpoint and last checkpoint
    last_checkpoint = dir_list[-1].split('/')[-2]
    try:
        trainer_state = json.load(open(model_path + 'trainer_state.json'))
        best_checkpoint = trainer_state["best_model_checkpoint"].split('/')[-1]
    except:
        # Load the last checkpoint's trainer state to retrieve the best checkpoint
        trainer_state = json.load(open(dir_list[-1] + 'trainer_state.json'))
        best_checkpoint = trainer_state["best_model_checkpoint"].split('/')[-1]
    
    for folder in dir_list:
        # Remove the other files
        if 'checkpoint' in folder and best_checkpoint not in folder and last_checkpoint not in folder:
            shutil.rmtree(folder)

def remove_redudant_checkpoints_opennmt(model_path):
    """
    Remove redundant checkpoints in the given model_path, keep only the best and the last checkpoint
    Warning: Deprecated, need to adjust for multirandom seeds
    """
    best_or_last_checkpoints = set()
    dir_list = glob(model_path + "/*")
    
    for file in dir_list:
        if '30000.pt' in file or 'best.pt' in file or '50000.pt' in file:
            best_or_last_checkpoints.add(file)
    # Add the last checkpoint
    best_or_last_checkpoints.add(dir_list[-1])

    for file in dir_list:
        # Remove the other files
        if file not in best_or_last_checkpoints and '_step_' in file:
            os.remove(file)

def main():
    dataset = 'NACS'
    # Loop through the HF model files
    model_dir_list = glob(os.getenv('BASE_DIR') + "/trained_models/" + dataset + "/*/")
    
    for model_dir in tqdm(model_dir_list):
        if 't5' in model_dir:
            # pdb.set_trace()
            remove_redudant_checkpoints_hf(model_dir)

    # Loop through the OpenNMT model files
    dataset = 'NACS'
    splits = {
        'standard', 'random', 'template', 'length', 'simple', 'tmcd',
        'random_str', 'random_cvcv', 'random_cvcv_shorter', 'random_str_shorter'
        'addprim_jump', 'addprim_turn_left', 'template_around_right',
        'turn_left_random_cvcv', 'turn_left_random_str', 'jump_random_cvcv', 
        'jump_random_str', 'no_mod', 'add_jump', 'add_turn_left'
    }
    model_dir_list = glob(os.getenv('BASE_DIR') + "/trained_models/" + dataset + "/*")
    for model_dir in tqdm(model_dir_list):
        if model_dir.split('/')[-1] in splits or 'HP/' in model_dir:
            dir_list = glob(model_dir + "/*/")
            for model_path in dir_list:
                if 'lstm' in model_path or 'transformer' in model_path:
                    remove_redudant_checkpoints_opennmt(model_path)


if __name__ == "__main__":
    main()