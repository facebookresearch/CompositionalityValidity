#!/bin/bash
# Generate prediction for T5
export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='geoquery'
export split='tmcd_random_str'
export lr='1e-4'
export batch_size='1'
export seed='12345'
export model_name='t5-base'

mkdir -p $base_dir/preds/$dataset_name/$split
if [[ ! $# -eq 0 ]];
then
  export dataset_name=$1
  export split=$2
  export model_name='t5-base'
fi

# load_best_model_at_end set to False because of TMCD
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
fi
cd $base_dir
mkdir -p ${base_dir}/logs/eval/${dataset_name}/

conda activate compgen

python hf_training/fine_tune_t5.py \
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
    python hf_training/fine_tune_t5.py \
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
        --load_best_model_at_end False \
        --metric_for_best_model "exact_match" \
        --evaluation_strategy "epoch" \
        --generation_num_beams 20 \
        --generation_max_length 256 \
        --output_dir "$model_dir/$dataset_name/${dir_model_name}_s{seed}_${split}_${lr}/"
fi
