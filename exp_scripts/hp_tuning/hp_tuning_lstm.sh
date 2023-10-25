#!/bin/bash

export dataset="pcfgset_hp"
# export split="simple"
export model_name="lstm_uni"   # Model name \in {lstm_uni, lstm_bi}

export base_dir=${BASE_DIR}
export input_path="${base_dir}/data/${dataset}/"
export data_output_path="${base_dir}/baseline_replication/COGS/processed_data/${dataset}/"
export TEST_SPLIT="test"
cd ${base_dir}/baseline_replication/COGS

export EXAMPLES=1_example  
############### Preparing for data ################
## Reformat data for NQG
python scripts/reformat_nqg_data_for_opennmt.py --input_path ${input_path} --output_path ${data_output_path}

export OPENNMT_DIR="${base_dir}/baseline_replication/COGS/src/OpenNMT-py"
## Preprocess data into OpenNMT format
python $OPENNMT_DIR/preprocess.py \
    -train_src $data_output_path/train_source.txt   \
    -train_tgt $data_output_path/train_target.txt   \
    -save_data $data_output_path/$EXAMPLES  \
    -src_seq_length 5000 -tgt_seq_length 5000   \
    -src_vocab $data_output_path/source_vocab.txt -tgt_vocab $data_output_path/target_vocab.txt


############### Model Training and Inference ################
                    # Number of exposure examples (1 or 100)
export SAVE_PATH=${base_dir}/trained_models/HP/${dataset} # Save path for checkpoints
export LOG_PATH=${base_dir}/logs           # Log path
export PRED_PATH=${base_dir}/baseline_replication/COGS/preds         # Predictions path
export SEED=1                                  # Random seed
export CUDA_VISIBLE_DEVICES=0                  # GPU machine number

if [[ $model_name == *"lstm_uni"* ]]
then
    encoder_type="rnn"
elif [[ $model_name == *"lstm_bi"* ]]
then
    encoder_type="brnn"
fi
mkdir -p $SAVE_PATH
mkdir -p ${base_dir}/logs/HP/${dataset}

declare -a dropouts=(0.0 0.1 0.5)
declare -a layers=(1 2)

if [[ $dataset == *"spider"* ]]
then
    gpus=3
else
    gpus=1
fi

# Loop for each dropout and layer
for layer in "${layers[@]}"
do
    for dropout in "${dropouts[@]}"
    do  
        SAVE_NAME=${EXAMPLES}_${model_name}_dr${dropout}_${layer}layer          # Checkpoint name

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${dataset}-${model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=${gpus}
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=$base_dir
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/HP/${dataset}/${SAVE_NAME}_output.log
#SBATCH --error=${base_dir}/logs/HP/${dataset}/${SAVE_NAME}_error.log
module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir/baseline_replication/COGS
conda activate compgen

        ## Training
        python -u $OPENNMT_DIR/train.py -data ${data_output_path}/$EXAMPLES -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
            -layers ${layer} -rnn_size 512 -word_vec_size 512 \
            -encoder_type $encoder_type -decoder_type rnn -rnn_type LSTM \
                -global_attention dot \
            -train_steps 50000  -max_generator_batches 2 -dropout ${dropout} \
            -batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
            -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 0.001 \
            -max_grad_norm 5 -param_init 0  \
            -valid_steps 500 -save_checkpoint_steps 500 \
            -early_stopping 5 --early_stopping_criteria loss \
            -world_size 1 -gpu_ranks 0 -seed $SEED --log_file ${LOG_PATH}/${dataset}_${SAVE_NAME}_s${SEED}.log 
            
        ## Inference
        echo SPLIT is $TEST_SPLIT
        if test -f "$SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt"; then
            python $OPENNMT_DIR/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                            -src ${data_output_path}/${TEST_SPLIT}_source.txt \
                                            -tgt ${data_output_path}/${TEST_SPLIT}_target.txt \
                                            -output ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                                            -replace_unk -verbose -shard_size 0 \
                                            -batch_size 128 \
                                            --max_length 2000
            paste ${data_output_path}/${TEST_SPLIT}_source.txt ${data_output_path}/${TEST_SPLIT}_target.txt ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
        else
            python $OPENNMT_DIR/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_step_50000.pt \
                                            -src ${data_output_path}/${TEST_SPLIT}_source.txt \
                                            -tgt ${data_output_path}/${TEST_SPLIT}_target.txt \
                                            -output ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                                            -replace_unk -verbose -shard_size 0 \
                                            -batch_size 128 \
                                            --max_length 2000
            paste ${data_output_path}/${TEST_SPLIT}_source.txt ${data_output_path}/${TEST_SPLIT}_target.txt ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ${PRED_PATH}/${TEST_SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
        fi
EOT
    done
done