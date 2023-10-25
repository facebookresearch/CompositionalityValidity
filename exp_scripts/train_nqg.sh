#!/bin/bash

export base_dir=${BASE_DIR}
export model_dir=$base_dir/trained_models
export data_dir=${base_dir}/data
export dataset_name='spider'
export split='template'
export SEED='12345'

echo "Training NQG with BASE_DIR=$base_dir"
## NQG specific params
export TRAIN_TSV="${data_dir}/${dataset_name}/${split}_split/train.tsv"
export TEST_TSV="${data_dir}/${dataset_name}/${split}_split/test.tsv"
export RULES="${data_dir}/${dataset_name}/${split}_split/rules.txt"
export BERT_DIR="${model_dir}/BERT/"
export TF_EXAMPLES="${data_dir}/${dataset_name}/${split}_split/tf_samples"
export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${model_dir}/${dataset_name}/nqg_${split}_${SEED}"

export NQG_TRAIN_TSV="${data_dir}/${dataset_name}/${split}_split/nqg_train.tsv"
export NQG_TEST_TSV="${data_dir}/${dataset_name}/${split}_split/nqg_test.tsv"

if [[ $dataset_name == *"geoquery"* ]]
then
    export sample_size=0
    export terminal_codelength=8
    export allow_repeated_target_nts=false
    export target_grammar="${base_dir}/baseline_replication/TMCD/model/parser/inference/targets/funql.txt"
    if [[ $split == *"template"* ]]
    then
      export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/geoquery_xl_config.json"
    fi
elif [[ $dataset_name == *"SCAN"* ]]
then
    export sample_size=500
    export terminal_codelength=32
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/scan_config.json"
elif [[ $dataset_name == *"spider"* ]]
then
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
elif [[ $dataset_name == *"COGS"* ]]
then
    export COGS_TRAIN_TSV=$TRAIN_TSV
    export COGS_TEST_TSV=$TEST_TSV
    export COGS_GEN_TSV="${data_dir}/${dataset_name}/${split}_split/gen.tsv"
    export NQG_GEN_TSV="${data_dir}/${dataset_name}/${split}_split/nqg_gen.tsv"
    TRAIN_TSV=$NQG_TRAIN_TSV
    TEST_TSV=$NQG_TEST_TSV
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/COGS_config.json"
fi


cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=nqg-${dataset_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=ALL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset_name}/nqg_${split}_s${SEED}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/nqg_${split}_s${SEED}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
conda activate compgen

# Convert the COGS formt into NQG format
if [[ $dataset_name == *"COGS"* ]]
then
  cd $base_dir/baseline_replication/COGS
  echo "Converting Train File"
  python convert_to_nqg_format.py --tsv=$COGS_TRAIN_TSV --output=$TRAIN_TSV
  echo "Converted Test File"
  python convert_to_nqg_format.py --tsv=$COGS_TEST_TSV --output=$TEST_TSV
  echo "Converted Gen File"
  python convert_to_nqg_format.py --tsv=$COGS_GEN_TSV --output=$NQG_GEN_TSV
fi

cd $base_dir/baseline_replication/TMCD

if [[ $dataset_name == *"spider"* ]]
then
    # Run the converting command
    # Note that for Spider, the script tasks/spider/nqg_preprocess.py should be run on the dataset TSV file to prepare the input for the space separated tokenization used by NQG.
    python tasks/spider/nqg_preprocess.py --input=$TRAIN_TSV --output=$NQG_TRAIN_TSV
    python tasks/spider/nqg_preprocess.py --input=$TEST_TSV --output=$NQG_TEST_TSV

    echo "Finish preprocessing data, starting to induce rules. With sample_size=${sample_size}, terminal_codelength=${terminal_codelength}, and allow_repeated_target_nts=${allow_repeated_target_nts}, input=${NQG_TRAIN_TSV}"

    python model/induction/induce_rules.py  \
      --input=${NQG_TRAIN_TSV} \
      --output=${RULES} \
      --sample_size=${sample_size} \
      --terminal_codelength=${terminal_codelength} \
      --allow_repeated_target_nts=${allow_repeated_target_nts}

    echo "Starting to write TF examples"
    python model/parser/data/write_examples.py \
      --input=${NQG_TRAIN_TSV} \
      --output=${TF_EXAMPLES} \
      --config=${CONFIG} \
      --rules=${RULES} \
      --bert_dir=${BERT_DIR}
else
    python model/induction/induce_rules.py  \
      --input=${TRAIN_TSV} \
      --output=${RULES} \
      --sample_size=${sample_size} \
      --terminal_codelength=${terminal_codelength} \
      --allow_repeated_target_nts=${allow_repeated_target_nts}

    python model/parser/data/write_examples.py \
      --input=${TRAIN_TSV} \
      --output=${TF_EXAMPLES} \
      --config=${CONFIG} \
      --rules=${RULES} \
      --bert_dir=${BERT_DIR}
fi

python model/parser/training/train_model.py \
  --input=${TF_EXAMPLES} \
  --config=${CONFIG} \
  --model_dir=${MODEL_DIR} \
  --bert_dir=${BERT_DIR} \
  --init_bert_checkpoint=False \
  --use_gpu \
  --seed=${SEED}

if [[ $dataset_name == *"spider"* ]]
then
    python model/parser/inference/eval_model.py \
      --input=${NQG_TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}
else
    python model/parser/inference/eval_model.py \
      --input=${TEST_TSV}  \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}
      --target_grammar=${target_grammar}  \
      --rules=${RULES}
fi
    
EOT