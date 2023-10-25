#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='geoquery'
export split='tmcd'
export seed='42'

## NQG specific params
export TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/test.tsv"

export RULES="${base_dir}/data/${dataset_name}/${split}_split/rules.txt"
export BERT_DIR="${base_dir}/trained_models/BERT/"
export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${base_dir}/trained_models/${dataset_name}/nqg_${split}_${seed}"

export NQG_TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_test.tsv"

export OUTPUT="${base_dir}/preds/${dataset_name}/${split}/"
export SOURCE_TXT="${base_dir}/data/${dataset_name}/${split}_split/source_test.txt"

if [[ $dataset_name == *"geoquery"* ]]
then
    export sample_size=0
    export terminal_codelength=8
    export allow_repeated_target_nts=false
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
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/COGS_config.json"
    export GEN_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_gen.tsv"
    export GEN_SOURCE_TXT="${data_dir}/${dataset_name}/${split}_split/source_gen.txt"
    TEST_TSV=$NQG_TEST_TSV
fi

mkdir -p $OUTPUT

cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=${dataset_name}-nqg
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
#SBATCH --output=${base_dir}/logs/${dataset_name}/eval_nqg_${split}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/eval_nqg_${split}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir/baseline_replication/TMCD
conda activate compgen


if [[ $dataset_name == *"spider"* ]]
then
    # Generate TXT file for sources
    python tasks/strip_targets.py \
        --input=${NQG_TEST_TSV} \
        --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${NQG_TEST_TSV}  \
      --output=${OUTPUT}/nqg_test_s${seed}.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}
else
    # Generate TXT file for sources
    python tasks/strip_targets.py \
        --input=${TEST_TSV} \
        --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${TEST_TSV}  \
      --output=${OUTPUT}/nqg_test_s${seed}.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}
fi

if [[ $dataset_name == *"COGS"* ]]
then
    # Generate TXT file for sources
    python tasks/strip_targets.py \
        --input=${GEN_TSV} \
        --output=${GEN_SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${GEN_TSV}  \
      --output=${OUTPUT}/nqg_gen_s${seed}.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}
fi
    
EOT
