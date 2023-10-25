#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved

export base_dir=${BASE_DIR}
export data_dir=$base_dir/data
export model_name='transformer'
export dir_model_name=$model_name
# for seed in 0 42 12345
for seed in 0 1 5 42 12345
do
    for split in random length tmcd template
    do

        pred_file_name=test_pred_1_example_${model_name}_s${seed}.txt

        if [[ $model_name == *"t5"* ]]
        then
            dir_model_name="t5-base"
            pred_file_name=${dir_model_name}_s${seed}_test.txt
        elif [[ $model_name == *"bart"* ]]
        then
            dir_model_name="bart-base"
            pred_file_name=${dir_model_name}_s${seed}_test.txt
        elif [ $model_name == *"lstm"* ] || [ $model_name == *"transformer"* ]
        then
            pred_file_name=test_pred_1_example_${model_name}_s${seed}.txt
        elif [[ $model_name == *"btg"* ]]
        then
            pred_file_name=${model_name}_${seed}_test.txt
        fi
        cd $base_dir

        mkdir -p ${base_dir}/results/spider_res/${split}
        sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pred-${dataset_name}-${dir_model_name}-${split}-s${seed}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=0-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${base_dir}
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=${base_dir}/logs/eval/${dataset_name}/${dir_model_name}_${split}_output_s${seed}.log
#SBATCH --error=${base_dir}/logs/eval/${dataset_name}/${dir_model_name}_${split}_error_s${seed}.log

        module load cuda
        module load anaconda3/2022.05
        source "/private/home/kaisersun/.bashrc"
        cd $base_dir
        conda activate compgen

        # # python ${base_dir}/baseline_replication/TMCD/tasks/spider/generate_gold.py --input="${data_dir}/spider/${split}_split/test.tsv" --output="${data_dir}/spider/${split}_split/target_test.txt"


        # if [[ $model_name == *"bart"* ]] || [[ $model_name == *"t5-base"* ]]
        # then
        #     # Remove pad and /s
        #     python ${base_dir}/baseline_replication/TMCD/tasks/spider/restore_oov.py --input="${base_dir}/preds/spider/${split}/${dir_model_name}_s${seed}_test.txt" --output="${base_dir}/preds/spider/${split}/$pred_file_name"
        # fi


        python ${base_dir}/baseline_replication/TMCD/tasks/spider/evaluation.py --gold="${data_dir}/spider/${split}_split/target_test.txt" --pred="${base_dir}/preds/spider/${split}/$pred_file_name" --table="${data_dir}/orig_spider/tables.json" --etype="match" --db="${data_dir}/orig_spider/database/"

EOT

    done
done
