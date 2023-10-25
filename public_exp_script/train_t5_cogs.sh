#!/bin/bash
# The script to train T5 on COGS
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
conda activate compgen

python -u hf_training/run_translation.py \
    --benchmark COGS \
    --model_name_or_path $model_name \
    --use_pretrained_weights True \
    --output_dir "$model_dir/$dataset_name/${dir_model_name}_s${seed}_${split}/" \
    --do_train \
    --do_predict \
    --source_lang en \
    --target_lang en \
    --source_prefix "translate English to English: " \
    --train_file $data_dir/${dataset_name}/${split}_split/train.json \
    --validation_file $data_dir/${dataset_name}/${split}_split/test.json \
    --test_file $data_dir/${dataset_name}/${split}_split/test.json \
    --gen_conditions_file $data_dir/${dataset_name}/cogs_gen_conditions.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir \
    --save_steps 2500000000 \
    --max_target_length 1024 \
    --max_source_length 1024 \
    --num_train_epochs $EPOCHS \
    --seed $seed \
    --learning_rate 0.000015 \
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
