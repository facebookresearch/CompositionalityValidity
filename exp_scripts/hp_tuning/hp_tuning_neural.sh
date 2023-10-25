#!/bin/bash
# Script for tunning hyperparmeters (of BART)
export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export dataset_name='pcfgset_hp'
export data_dir=$base_dir/data
export lr='1e-5'
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
    batch_size='4'
fi

export WANDB_DISABLED=true
cd $base_dir

mkdir -p ${base_dir}/logs/HP/
mkdir -p $model_dir/HP/$dataset_name/

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=HP-${dataset_name}-${dir_model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=$base_dir
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/HP/${dataset_name}/${dir_model_name}_${lr}_output.log
#SBATCH --error=${base_dir}/logs/HP/${dataset_name}/${dir_model_name}_${lr}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir
conda activate compgen

mkdir -p "$model_dir/HP/$dataset_name/${dir_model_name}_${lr}/"

python hf_training/fine_tune_t5.py \
    --model_name_or_path $model_name \
    --train_file "$data_dir/${dataset_name}/train.csv" \
    --validation_file "$data_dir/${dataset_name}/test.csv" \
    --do_train \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --gradient_accumulation_steps 32 \
    --max_seq_length 512  \
    --max_output_length 256 \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "exact_match" \
    --evaluation_strategy "epoch" \
    --generation_num_beams 20 \
    --generation_max_length 512 \
    --output_dir "$model_dir/HP/$dataset_name/${dir_model_name}_${lr}/"
    
EOT