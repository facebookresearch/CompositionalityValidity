#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='spider'
export split='template'
export lr='1e-4'
export batch_size='1'
export seed='12345'
export model_name='facebook/bart-base'
mkdir -p $base_dir/preds/$dataset_name/$split
if [[ ! $# -eq 0 ]];
then
  export dataset_name=$1
  export split=$2
  export model_name='facebook/bart-base'
fi

# load_best_model_at_end set to False because of TMCD
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart-base"
fi
cd $base_dir
mkdir -p ${base_dir}/logs/eval/${dataset_name}/

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pred-${dataset_name}-${dir_model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/eval/${dataset_name}/${dir_model_name}_${split}_output_${lr}_s${seed}.log
#SBATCH --error=${base_dir}/logs/eval/${dataset_name}/${dir_model_name}_${split}_error_${lr}_s${seed}.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir
conda activate compgen

python hf_training/fine_tune_bart.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}_${lr}/" \
    --validation_file "$data_dir/$dataset_name/${split}_split/test.tsv" \
    --do_eval \
    --do_predict \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512  \
    --max_output_length 256 \
    --save_strategy "epoch" \
    --load_best_model_at_end False \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 256 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}_${lr}/"

if [[ $dataset_name == *"COGS"* ]]
then
    python hf_training/fine_tune_bart.py \
        --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}_${lr}/" \
        --validation_file "$data_dir/$dataset_name/${split}_split/gen.tsv" \
        --do_eval \
        --do_predict \
        --seed $seed \
        --predict_with_generate \
        --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps 16 \
        --max_seq_length 512  \
        --max_output_length 256 \
        --save_strategy "epoch" \
        --load_best_model_at_end True \
        --metric_for_best_model "loss" \
        --evaluation_strategy "epoch" \
        --generation_num_beams 20 \
        --generation_max_length 256 \
        --output_dir "$model_dir/$dataset_name/${dir_model_name}_s{seed}_${split}_${lr}/"
fi

EOT
