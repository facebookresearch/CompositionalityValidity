#!/bin/bash
# Train T5 script
export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='SCAN'
export split='jump_random_cvcv'
export lr='1e-4'
export batch_size='1'
export seed='0'
export epoch='60'
# Epoch = 400 for SPIDER, 850 for GEOQUERY, 60 for SCAN
export model_name='t5-base'
# load_best_model_at_end set to False because of TMCD

dir_model_name=$model_name

cd $base_dir
mkdir -p $model_dir/$dataset_name/

cd $base_dir
conda activate compgen

if [[ $dataset_name != *"COGS"* ]]
then
    python hf_training/fine_tune_t5.py \
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
else
    # Because COGS has a development set, we can select load_best_checkpoint as true at last
    python hf_training/fine_tune_t5.py \
        --model_name_or_path $model_name \
        --train_file "$data_dir/$dataset_name/${split}_split/nqg_train.tsv" \
        --validation_file "$data_dir/$dataset_name/${split}_split/nqg_dev.tsv" \
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
        --generation_num_beams 40 \
        --generation_max_length 256 \
        --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}_${lr}/"
fi
