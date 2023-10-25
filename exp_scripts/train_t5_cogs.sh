#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export dataset_name='COGS'
export data_dir=$base_dir/data
export split='length'
export seed=12345
export model_name='t5-base'
# export model_name='facebook/bart-base'
export EPOCHS=100
echo "Training with BASE_DIR=${base_dir}"

if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi
cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${split}-${dataset_name}-${dir_model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --gpus=4
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}/baseline_replication/TMCD
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_s${seed}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_s${seed}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir
conda activate compgen

# python -u hf_training/run_translation.py \
#     --benchmark COGS \
#     --model_name_or_path $model_name \
#     --use_pretrained_weights True \
#     --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
#     --do_train \
#     --do_predict \
#     --source_lang en \
#     --target_lang en \
#     --source_prefix "translate English to English: " \
#     --train_file $data_dir/${dataset_name}/${split}_split/train.json \
#     --validation_file $data_dir/${dataset_name}/${split}_split/test.json \
#     --test_file $data_dir/${dataset_name}/${split}_split/test.json \
#     --gen_conditions_file $data_dir/${dataset_name}/cogs_gen_conditions.txt \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --overwrite_output_dir \
#     --save_steps 2500000000 \
#     --max_target_length 1024 \
#     --max_source_length 1024 \
#     --num_train_epochs $EPOCHS \
#     --seed $seed \
#     --learning_rate 0.000015 \
#     --predict_with_generate

python -u hf_training/run_translation.py \
    --benchmark COGS \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
    --use_pretrained_weights True \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
    --do_predict \
    --source_lang en \
    --target_lang en \
    --source_prefix "translate English to English: " \
    --validation_file $data_dir/${dataset_name}/${split}_split/test.json \
    --test_file $data_dir/${dataset_name}/${split}_split/gen.json \
    --gen_conditions_file $data_dir/${dataset_name}/cogs_gen_conditions.txt \
    --per_device_eval_batch_size 32 \
    --max_target_length 1024 \
    --max_source_length 1024 \
    --num_train_epochs $EPOCHS \
    --seed $seed \
    --predict_with_generate

python -u hf_training/run_translation.py \
    --benchmark COGS \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
    --use_pretrained_weights True \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
    --do_predict \
    --source_lang en \
    --target_lang en \
    --source_prefix "translate English to English: " \
    --validation_file $data_dir/${dataset_name}/${split}_split/test.json \
    --test_file $data_dir/${dataset_name}/${split}_split/test.json \
    --gen_conditions_file $data_dir/${dataset_name}/cogs_gen_conditions.txt \
    --per_device_eval_batch_size 32 \
    --max_target_length 1024 \
    --max_source_length 1024 \
    --num_train_epochs $EPOCHS \
    --seed $seed \
    --predict_with_generate

EOT
