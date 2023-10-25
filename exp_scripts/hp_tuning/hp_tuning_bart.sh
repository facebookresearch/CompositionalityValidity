#!/bin/bash
# For tuning the hyperparameters of BART with AllenNLP

export base_dir=$BASE_DIR
export dataset_name='pcfgset_hp'
export seed='42'
export lr=5e-5
# 2e-4, 2e-5, 5e-5
export data_dir=$base_dir/data/${dataset_name}

export traindata_name=$data_dir'/train.tsv'
export testdata_name=$data_dir'/test.tsv'
export archive_dirname="${base_dir}/trained_models/HP/${dataset_name}/bart-base_${split}_s${seed}"

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=BART-hp-s${seed}-${dataset_name}
#SBATCH --mail-user=kaisersun@meta.com
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=$base_dir
#SBATCH --export=all
#SBATCH --constraint=volta32gb,ib4
#SBATCH --output=$base_dir/logs/HP/${dataset_name}/bart-base_${lr}_output.log
#SBATCH --error=$base_dir/logs/HP/${dataset_name}/bart-base_${lr}_error.log

# module load anaconda3
module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd ${BASE_DIR}/baseline_replication/seq2seqCOGS/
conda activate compgen

allennlp train \
            'configs/BART_other.jsonnet' \
            --serialization-dir  $archive_dirname\
            --include-package modules \
            -f --file-friendly-logging \
            -o '{"random_seed": '$seed', "numpy_seed": '$seed', "pytorch_seed": '$seed',
                 "train_data_path": "'$traindata_name'",
                 "validation_data_path": "'$traindata_name'",
                 "test_data_path": "'$testdata_name'",
                 "data_loader.batch_sampler.batch_size": '4',
               #   "distributed": True,
                 "trainer.cuda_device": 0,
                 "trainer.optimizer.lr": $lr,
                 "trainer.num_gradient_accumulation_steps": '32'
                 }'


EOT