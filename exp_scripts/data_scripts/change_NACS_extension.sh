export base_dir=$BASE_DIR
export dataset=pcfgset
# Change the directory name into split_name + '_split'
for dir in ${base_dir}/data/${dataset}/*/ ; do
    new_dir="${dir::-1}_split"
    echo $new_dir
    mv $dir $new_dir
done

# Change the data extension of NACS dataset to be txt and unify it for BTG
for dir in ${base_dir}/data/${dataset}/*/ ; do
    echo $dir
    cd $dir
    mv train.src btg_source_train.txt
    mv train.trg btg_target_train.txt
    mv test.src btg_source_test.txt
    mv test.trg btg_target_test.txt
done

# Convert it into HF Readable format
for dir in ${base_dir}/data/${dataset}/*/ ; do
    python $base_dir/baseline_replication/TMCD/tasks/scan/join_txt_to_tsv.py --source=${dir}btg_source_train.txt --target=${dir}btg_target_train.txt --output=${dir}train.tsv
    python $base_dir/baseline_replication/TMCD/tasks/scan/join_txt_to_tsv.py --source=${dir}btg_source_test.txt --target=${dir}btg_target_test.txt --output=${dir}test.tsv

done

# Create corresponding hyperparameter tuning sets
python split_dataset_for_hp.py --input=${base_dir}/data/${dataset}/simple_split --dataset=${dataset}