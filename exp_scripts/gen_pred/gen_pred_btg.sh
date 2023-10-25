#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
export dataset="geoquery"
export split="tmcd_random_str"
export seed=12345
export base_dir=${BASE_DIR}
export input_path="${base_dir}/data/${dataset}/${split}"
export TEST_SPLIT="test"
export output_dir=${base_dir}/trained_models/${dataset}/${split}
export lr=1e-4
export gradient_accumulation_steps=8
export warmup_steps=0
export batch_size=25
export segment=3
# Use the resulting train_steps to determine the optimial step
export train_step=10000

export eval_step_list=$train_step
if [[ $dataset == *"geoquery"* ]]
then
    export vocab_size=200
    export lr=5e-5
    export max_source_length=64
    export max_target_length=64
    export train_step=15000
elif [[ $dataset == *"SCAN"* ]]
then
    export vocab_size=100
    export max_source_length=128
    export max_target_length=128
    export segment=1
elif [[ $dataset == *"spider"* ]]
then
    export vocab_size=800
    export max_source_length=128
    export max_target_length=128
    export gradient_accumulation_steps=32
    export batch_size=32
    export train_step=15000
    export full_train_step=15000
    export warmup=2000
    export segment=1
elif [[ $dataset == *"COGS"* ]]
then
    export vocab_size=400
    export lr=1e-4
    export max_source_length=64
    export max_target_length=64
    export train_step=15000
fi

export train_step=14400
SAVE_NAME=btg_$seed         # Checkpoint name
echo "Training for ${train_step} steps"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pred-${dataset}-${split}-btg-s${seed}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset}/${SAVE_NAME}_${split}_output.log
#SBATCH --error=${base_dir}/logs/${dataset}/${SAVE_NAME}_${split}_error.log
# module load anaconda3
module load cuda
module load anaconda3/2022.05
source "/private/home/kaisersun/.bashrc"

cd $base_dir/baseline_replication/btg-seq2seq
conda activate btg-seq2seq
which python
echo $CONDA_DEFAULT_ENV
    CUDA_VISIBLE_DEVICES=0	 python neural_btg/commands/train_nmt_seg2seg.py \
        --do_eval \
        --do_test \
        --use_posterior \
        --seed $seed \
        --eval_step_list $eval_step_list \
        --dev_filename ${base_dir}/data/${dataset}/${split}_split/btg_source_test.txt,${base_dir}/data/${dataset}/${split}_split/btg_target_test.txt \
        --test_filename ${base_dir}/data/${dataset}/${split}_split/btg_source_test.txt,${base_dir}/data/${dataset}/${split}_split/btg_target_test.txt \
        --load_model_path $output_dir/$SAVE_NAME/step_checkpoints/$train_step.bin \
        --output_dir $output_dir/$SAVE_NAME \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --vocab_size $vocab_size \
        --num_segments $segment \
        --geo_p 0.6 \
        --num_segre_samples 1 \
        --max_source_segment_length 4 \
        --max_target_segment_length 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps  \
        --beam_size 10 \
        --train_batch_size ${batch_size} \
        --eval_batch_size ${batch_size} \
        --learning_rate $lr \
        --warmup_steps $warmup_steps \
        --train_steps $train_step \
        --eval_steps 600

    # Copy the prediction file to pred
    cp $output_dir/$SAVE_NAME/final.output ${BASE_DIR}/preds/${dataset}/${split}/btg_${seed}_test.txt

    # CUDA_VISIBLE_DEVICES=0	 python neural_btg/commands/train_nmt_seg2seg.py \
    #     --do_eval \
    #     --do_test \
    #     --use_posterior \
    #     --seed $seed \
    #     --eval_step_list $eval_step_list \
    #     --dev_filename ${base_dir}/data/${dataset}/${split}_split/btg_source_gen.txt,${base_dir}/data/${dataset}/${split}_split/btg_target_gen.txt \
    #     --test_filename ${base_dir}/data/${dataset}/${split}_split/btg_source_gen.txt,${base_dir}/data/${dataset}/${split}_split/btg_target_gen.txt \
    #     --load_model_path $output_dir/$SAVE_NAME/step_checkpoints/$train_step.bin \
    #     --output_dir $output_dir/$SAVE_NAME \
    #     --max_source_length $max_source_length \
    #     --max_target_length $max_target_length \
    #     --vocab_size $vocab_size \
    #     --num_segments $segment \
    #     --geo_p 0.6 \
    #     --num_segre_samples 1 \
    #     --max_source_segment_length 4 \
    #     --max_target_segment_length 4 \
    #     --gradient_accumulation_steps $gradient_accumulation_steps  \
    #     --beam_size 10 \
    #     --train_batch_size ${batch_size} \
    #     --eval_batch_size ${batch_size} \
    #     --learning_rate $lr \
    #     --warmup_steps $warmup_steps \
    #     --train_steps $train_step \
    #     --eval_steps 600

    # # Copy the prediction file to pred
    # cp $output_dir/$SAVE_NAME/final.output ${BASE_DIR}/preds/${dataset}/${split}/btg_${seed}_gen.txt
EOT
