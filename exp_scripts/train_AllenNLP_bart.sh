#!/bin/bash

export base_dir=$BASE_DIR
export dataset_name='NACS'
export split='add_turn_left'
export seed='42'
export data_dir=$base_dir/data/${dataset_name}/${split}_split
export pred_dir=$base_dir/preds/${dataset_name}/${split}

export traindata_name=$data_dir'/train.tsv'
export testdata_name=$data_dir'/test.tsv'
export testoutput_filename="$pred_dir/bart-base_s${seed}_test"
export archive_dirname="${base_dir}/trained_models/${dataset_name}/bart-base_${split}_s${seed}"

if [[ $dataset_name == *"NACS"* ]]
then
    lr=2e-5
else
    lr=2e-4
fi

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=BART-${split}-s${seed}-${dataset_name}
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
#SBATCH --output=$base_dir/logs/${dataset_name}/bart-base_s${seed}_${split}_output.log
#SBATCH --error=$base_dir/logs/${dataset_name}/bart-base_s${seed}_${split}_error.log

# module load anaconda3
module load anaconda3/2022.05
module load cuda
source "/private/home/kaisersun/.bashrc"
cd ${BASE_DIR}/baseline_replication/seq2seqCOGS/
conda activate compgen

# allennlp train \
#             'configs/BART_other.jsonnet' \
#             --serialization-dir  $archive_dirname\
#             --include-package modules \
#             -f --file-friendly-logging \
#             -o '{"random_seed": '$seed', "numpy_seed": '$seed', "pytorch_seed": '$seed',
#                  "train_data_path": "'$traindata_name'",
#                  "validation_data_path": "'$traindata_name'",
#                  "test_data_path": "'$testdata_name'",
#                  "data_loader.batch_sampler.batch_size": '4',
#                #   "distributed": True,
#                  "trainer.cuda_device": 0,
#                  "trainer.optimizer.lr": ${lr},
#                  "trainer.num_gradient_accumulation_steps": '32'
#                  }'


mkdir -p $pred_dir

echo "Evaluating model on $testdata_name..."

allennlp predict \
          $archive_dirname \
          $testdata_name \
          --batch-size 16 \
          --silent \
          --use-dataset-reader \
          --include-package modules \
          --cuda-device 0 \
          --output-file $testoutput_filename

python scripts/json2txt.py $testoutput_filename $testdata_name  $testoutput_filename'.txt'

python scripts/eval.py $testoutput_filename'.txt'

echo "Done. Output saved in "${pred_dir}/""


EOT