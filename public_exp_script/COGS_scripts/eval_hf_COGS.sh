#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
# Evaluate T5 on COGS test set and generalization set
export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export dataset_name='COGS'
export data_dir=$base_dir/data
export lr='1e-4'
export batch_size='1'
export seed='42'
export epoch='20'
export model_name='t5-base'

if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi

cd $base_dir

conda activate compgen

python hf_training/fine_tune_t5.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_${lr}/" \
    --validation_file "$data_dir/${dataset_name}/standard_split/test.csv" \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512  \
    --max_output_length 512 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 512 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${lr}/"

python hf_training/fine_tune_t5.py \
    --model_name_or_path "$model_dir/$dataset_name/${dir_model_name}_${lr}/" \
    --validation_file "$data_dir/${dataset_name}/standard_split/gen.csv" \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512  \
    --max_output_length 512 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 512 \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_${lr}/"
