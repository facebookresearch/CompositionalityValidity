#!/bin/bash
# Evaluate the ensemble, NQG-T5
export base_dir=$BASE_DIR
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data
export dataset_name='spider'
export split='random'
export t5_seed='0'
export eval_split='test'
export pred_dir=${base_dir}/preds

## NQG specific params
export TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/test.tsv"
export RULES="${base_dir}/data/${dataset_name}/${split}_split/rules.txt"
export BERT_DIR="${base_dir}/trained_models/BERT/"
export TF_EXAMPLES="${base_dir}/data/${dataset_name}/${split}_split/tf_samples"
export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/${dataset_name}_config.json"
export MODEL_DIR="${base_dir}/trained_models/${dataset_name}/nqg_${split}"

export NQG_TEST_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_test.tsv"

export T5_pred="${pred_dir}/${dataset_name}/split/t5_s${t5_seed}_${eval_split}.txt"
export OUTPUT="${base_dir}/preds/${dataset_name}/${split}"

if [[ $dataset_name == *"SCAN"* ]]
then
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/scan_config.json"
elif [[ $dataset_name == *"spider"* ]]
then
    export T5_pred="${pred_dir}/${dataset_name}/${split}/t5-base_s${t5_seed}_cleaned.txt"
elif [[ $dataset_name == *"COGS"* ]]
then
    export GEN_TSV="${base_dir}/data/${dataset_name}/${split}_split/nqg_gen.tsv"
    export GEN_SOURCE_TXT="${data_dir}/${dataset_name}/${split}_split/source_gen.txt"
    TEST_TSV=$NQG_TEST_TSV
    export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/COGS_config.json"
elif [[ $dataset_name == *"geoquery"* ]]
then
    export target_grammar="${base_dir}/baseline_replication/TMCD/model/parser/inference/targets/funql.txt"
    if [[ $split == *"template"* ]]
    then
        export CONFIG="${base_dir}/baseline_replication/TMCD/model/parser/configs/geoquery_xl_config.json"
    fi
fi


cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=evalnqgt5-${dataset_name}
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
#SBATCH --output=${base_dir}/logs/${dataset_name}/nqgt5_${split}_output.log
#SBATCH --error=${base_dir}/logs/${dataset_name}/nqgt5_${split}_error.log

module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd $base_dir/baseline_replication/TMCD
conda activate compgen

# Check if corresponding T5 prediction exists, if not run the command go generate T5 predictions

# if [ ! -f ${t5_pred}]
# then
#   # Generate T5 prediction
#   bash scripts/gen_pred_hf.sh ${dataset_name} ${split}
# fi

if [[ $dataset_name == *"spider"* ]]
then
    # python model/parser/inference/eval_model.py \
    #   --input=${NQG_TEST_TSV}  \
    #   --config=${CONFIG}   \
    #   --model_dir=${MODEL_DIR}   \
    #   --bert_dir=${BERT_DIR}   \
    #   --target_grammar=${target_grammar}  \
    #   --rules=${RULES} \
    #   --fallback_predictions=${T5_pred}

    # # Generate TXT file for sources
    # python tasks/strip_targets.py \
    #     --input=${NQG_TEST_TSV} \
    #     --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${NQG_TEST_TSV}  \
      --output=${OUTPUT}/nqgt5_test_pred.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --target_grammar=${target_grammar}  \
      --rules=${RULES} \
      --fallback_predictions=${T5_pred}
else
    # python model/parser/inference/eval_model.py \
    #   --input=${TEST_TSV}  \
    #   --config=${CONFIG}   \
    #   --model_dir=${MODEL_DIR}   \
    #   --bert_dir=${BERT_DIR}   \
    #   --rules=${RULES} \
    #   --target_grammar=${target_grammar}  \
    #   --fallback_predictions=${T5_pred}

    # # Generate TXT file for sources
    # python tasks/strip_targets.py \
    #     --input=${TEST_TSV} \
    #     --output=${SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${TEST_TSV}  \
      --output=${OUTPUT}/nqgt5_test_pred.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}  \
      --target_grammar=${target_grammar}  \
      --fallback_predictions=${T5_pred}
fi

if [[ $dataset_name == *"COGS"* ]]
then
    # # Generate TXT file for sources
    # python tasks/strip_targets.py \
    #     --input=${GEN_TSV} \
    #     --output=${GEN_SOURCE_TXT}

    python model/parser/inference/generate_predictions.py \
      --input=${GEN_TSV}  \
      --output=${OUTPUT}/nqgt5_gen_pred.txt \
      --config=${CONFIG}   \
      --model_dir=${MODEL_DIR}   \
      --bert_dir=${BERT_DIR}   \
      --rules=${RULES}  \
      --target_grammar=${target_grammar}  \
      --fallback_predictions=${T5_pred}
fi
    
EOT