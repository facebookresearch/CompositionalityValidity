#!/bin/bash
# Train BART with AllenNLP repo (for COGS, GeoQuery, and SCAN)
export base_dir=$BASE_DIR
export dataset_name='SCAN'
export split='turn_left_random_str'
export seed='42'
export data_dir=$base_dir/data/${dataset_name}/${split}_split
export pred_dir=$base_dir/preds/${dataset_name}/${split}

export traindata_name=$data_dir'/train.tsv'
export testdata_name=$data_dir'/test.tsv'
export testoutput_filename="$pred_dir/bart-base_s${seed}_test"
export archive_dirname="${base_dir}/trained_models/${dataset_name}/bart-base_${split}_s${seed}"

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
                 "trainer.optimizer.lr": '2e-4',
                 "trainer.num_gradient_accumulation_steps": '32'
                 }'


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

# python scripts/eval.py $testoutput_filename'.txt'

echo "Done. Output saved in "${pred_dir}/""
