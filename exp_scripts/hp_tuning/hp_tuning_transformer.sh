#!/bin/bash

export dataset="pcfgset_hp"
export model_name="transformer"   # Model name \in {transformer}

export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)

export base_dir=${BASE_DIR}
export input_path="${base_dir}/data/${dataset}/${split}_split"
export data_output_path="${base_dir}/baseline_replication/COGS/processed_data/${dataset}/${split}"
export TEST_SPLIT="test"
cd ${base_dir}/baseline_replication/COGS
export OPENNMT_DIR="${base_dir}/baseline_replication/COGS/src/OpenNMT-py"

############### Preparing for data ################
# If no previous preprocessing is done, then prepare data
if [ -z "$(ls -A ${data_output_path})" ]; then
   ## Reformat data for NQG
    python scripts/reformat_nqg_data_for_opennmt.py --input_path ${input_path} --output_path ${data_output_path}
    ## Preprocess data into OpenNMT format
    python $OPENNMT_DIR/preprocess.py \
        -train_src $data_output_path/train_source.txt   \
        -train_tgt $data_output_path/train_target.txt   \
        -save_data $data_output_path/$EXAMPLES  \
        -src_seq_length 5000 -tgt_seq_length 5000   \
        -src_vocab $data_output_path/source_vocab.txt -tgt_vocab $data_output_path/target_vocab.txt
fi


############### Model Training and Inference ################
export SAVE_PATH=${base_dir}/trained_models/HP/${dataset}  # Save path for checkpoints
# export SAVE_NAME=${EXAMPLES}_${model_name}          # Checkpoint name
export LOG_PATH=${base_dir}/logs           # Log path
export PRED_PATH=${base_dir}/baseline_replication/COGS/preds         # Predictions path
export SEED=1                                  # Random seed
export CUDA_VISIBLE_DEVICES=0                  # GPU machine number

encoder_type="transformer"

mkdir -p $SAVE_PATH
mkdir -p ${base_dir}/logs/HP/${dataset}


declare -a dropouts=(0.0 0.1 0.5)
declare -a layers=(2 4)
# declare -a layers=(6)

# Loop for each dropout and layer
for layer in "${layers[@]}"
do
    for dropout in "${dropouts[@]}"
    do  
        SAVE_NAME=${EXAMPLES}_${model_name}_dr${dropout}_${layer}layer          # Checkpoint name
        
        if [[ $layer == *"8"* ]]
        then
            gpus=4
        else
            gpus=1
        fi

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${dataset}-${model_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=$gpus
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/private/home/kaisersun/CompGenComparision
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
	-layers ${layer} -rnn_size 512 -word_vec_size 512 -transformer_ff 512 -heads 4  \
	-encoder_type $model_name -decoder_type $model_name -position_encoding \
	-train_steps 30000  -max_generator_batches 2 -dropout ${dropout} \
	-batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
	-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
	-max_grad_norm 0 -param_init 0  -param_init_glorot \
	-label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 \
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
    python $OPENNMT_DIR/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_step_30000.pt \
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