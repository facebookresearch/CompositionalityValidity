#!/bin/bash

export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='spider'
export split='length'
export lr='1e-4'
export batch_size='1'
export seed='12345'
export epoch='200'
# Epoch = 400 for SPIDER, 850 for GEOQUERY, 60 for SCAN
export model_name='facebook/bart-base'
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart-base"
elif [[ $model_name == *"bloom"* ]]
then
    dir_model_name="bloom"
elif [[ $model_name == *"bigbird"* ]]
then
    dir_model_name="bigbird"
fi
cd $base_dir
mkdir -p $model_dir/$dataset_name/

python hf_training/fine_tune_bart.py \
        --model_name_or_path $model_name \
        --train_file "$data_dir/$dataset_name/${split}_split/train.tsv" \
        --validation_file "$data_dir/$dataset_name/${split}_split/test.tsv" \
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
        --load_best_model_at_end False \
        --metric_for_best_model "exact_match" \
        --evaluation_strategy "epoch" \
        --generation_num_beams 20 \
        --generation_max_length 256 \
        --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}_${lr}/"
