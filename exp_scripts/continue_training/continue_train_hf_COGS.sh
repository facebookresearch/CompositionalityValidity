#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export dataset_name='COGS'
export data_dir=$base_dir/data
export split='standard'
export lr='1e-4'
export batch_size='1'
export seed='42'
export epoch='40'
export model_name='t5-base'

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

#SBATCH --job-name=cont-${dataset_name}-${dir_model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/private/home/kaisersun/CompGenComparision/baseline_replication/TMCD
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_${lr}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_${lr}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir
conda activate gen

python hf_training/fine_tune_t5.py \
    --model_name_or_path $model_name \
    --train_file "$data_dir/${dataset_name}/${split}_split/train.csv" \
    --validation_file "$data_dir/${dataset_name}/${split}_split/dev.csv" \
    --do_train \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512  \
    --max_output_length 256 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 256 \
    --resume_from_checkpoint "$model_dir/$dataset_name/${dir_model_name}_${split}_${lr}/checkpoint-4888" \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${split}_${lr}/"
    
EOT
