#!/bin/bash
# Potentially buggy, need to check
export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='geoquery'
export split='length'

## NQG specific params
export TRAIN_TSV="${base_dir}/data/${dataset_name}/${split}_split/train.tsv"
export TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/test.tsv"
export RULES="${base_dir}/data/${dataset_name}/${split}_split/rules.txt"
export BERT_DIR="${base_dir}/trained_models/BERT/"
export TF_EXAMPLES="${base_dir}/data/${dataset_name}/${split}_split/tf_samples"
export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${base_dir}/trained_models/${dataset_name}/nqg_${split}"

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
    export sample_size=1000
    export terminal_codelength=8
    export allow_repeated_target_nts=true
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/COGS_config.json"
fi
# Note that for Spider, the script tasks/spider/nqg_preprocess.py should be run on the dataset TSV file to prepare the input for the space separated tokenization used by NQG.

cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=cont-${dataset_name}-nqg
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=ALL
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --time=0-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/${dataset_name}/nqg_${split}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/nqg_${split}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir
conda activate compgen


python ${base_dir}/baseline_replication/TMCD/model/parser/training/train_model.py \
  --input=${TF_EXAMPLES} \
  --config=${CONFIG} \
  --model_dir=${MODEL_DIR} \
  --bert_dir=${BERT_DIR} \
  --init_bert_checkpoint=False \
  --restore_checkpoint=True \
  --use_gpu
    
EOT