#!/bin/bash

export base_dir=/private/home/kaisersun/CompGenComparision
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='NACS'
export split='simple'
export lr='1e-4'
export batch_size='1'
export seed='12345'
export epoch='60'
# Epoch = 400 for SPIDER, 850 for GEOQUERY, 60 for SCAN
export model_name='t5-base'
# export model_name="twigs/bigbird-pegasus-large"
# load_best_model_at_end set to False because of TMCD
if [[ $model_name == *"t5"* ]]
then
    dir_model_name=$model_name
elif [[ $model_name == *"bart"* ]]
then
    dir_model_name="bart"
elif [[ $model_name == *"bloom"* ]]
then
    dir_model_name="bloom"
elif [[ $model_name == *"bigbird"* ]]
then
    dir_model_name="bigbird"
fi
cd $base_dir
mkdir -p $model_dir/$dataset_name/
mkdir -p ${base_dir}/logs/${dataset_name}

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${dataset_name}-${dir_model_name}-${split}-$seed
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_output_${lr}_s${seed}.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/${dir_model_name}_${split}_error_${lr}_s${seed}.log

# module load anaconda3
module load cuda
module load anaconda3/2022.05
source "/private/home/kaisersun/.bashrc"
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

EOT