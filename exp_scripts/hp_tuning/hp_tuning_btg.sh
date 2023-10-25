#!/bin/bash
export dataset="NACS_hp"

export base_dir=${BASE_DIR}
export input_path="${base_dir}/data/${dataset}/"
export TEST_SPLIT="test"
export output_dir=${base_dir}/trained_models/HP/${dataset}

export gradient_accumulation_steps=8
export batch_size=25

# Use the resulting train_steps to determine the optimial step
declare -a train_steps=(30000)
declare -a lrs=(1e-4 3e-4)
# declare -a lrs=(1e-5 5e-5)
if [[ $dataset == *"geoquery"* ]]
then
    declare -a vocab_sizes=(200 400)
    export max_source_length=64
    export max_target_length=64
elif [[ $dataset == *"SCAN"* ]]
then
    declare -a vocab_sizes=(25 50)
    export max_source_length=256
    export max_target_length=256
elif [[ $dataset == *"NACS"* ]]
then
    declare -a vocab_sizes=(25 50)
    export max_source_length=256
    export max_target_length=256
    declare -a train_steps=(5000)
elif [[ $dataset == *"spider"* ]]
then
    declare -a vocab_sizes=(200 400 800)
    export max_source_length=128
    export max_target_length=128
    export gradient_accumulation_steps=32
    export batch_size=32
else 
    # [[ $model_name == *"COGS"* ]]
    declare -a vocab_sizes=(200 400)
    export max_source_length=64
    export max_target_length=64
fi

mkdir -p ${base_dir}/logs/HP/${dataset}/
mkdir -p $output_dir

# Loop for each dropout and layer
for vocab_size in "${vocab_sizes[@]}"
do
    for train_step in "${train_steps[@]}"
    do
        for lr in "${lrs[@]}"
        do
            SAVE_NAME=btg_vs${vocab_size}_ts${train_step}_lr${lr}          # Checkpoint name
            echo $SAVE_NAME
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=hp-${dataset}-btg
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/HP/${dataset}/${SAVE_NAME}_output.log
#SBATCH --error=${base_dir}/logs/HP/${dataset}/${SAVE_NAME}_error.log
module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir/baseline_replication/btg-seq2seq
conda activate btg-seq2seq

    CUDA_VISIBLE_DEVICES=0	 python neural_btg/commands/train_nmt_seg2seg.py \
        --do_train \
        --do_eval \
        --use_posterior \
        --train_filename $base_dir/data/${dataset}/btg_source_train.txt,${base_dir}/data/${dataset}/btg_target_train.txt \
        --dev_filename ${base_dir}/data/${dataset}/btg_source_test.txt,${base_dir}/data/${dataset}/btg_target_test.txt \
        --output_dir $output_dir/$SAVE_NAME \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --vocab_size $vocab_size \
        --num_segments 3 \
        --geo_p 0.6 \
        --num_segre_samples 1 \
        --max_source_segment_length 4 \
        --max_target_segment_length 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps  \
        --beam_size 10 \
        --train_batch_size ${batch_size} \
        --eval_batch_size ${batch_size} \
        --learning_rate $lr \
        --warmup_steps 0 \
        --train_steps ${train_step} \
        --eval_steps 200
EOT
        done
    done
done