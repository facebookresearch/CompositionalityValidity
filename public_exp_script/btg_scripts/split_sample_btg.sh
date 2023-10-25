# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
export dataset="SCAN"
# declare -a splits=('standard_random_cvcv' 'standard_random_str' 'tmcd_random_cvcv' 'tmcd_random_str')
# declare -a splits=('length' 'random' 'tmcd' 'template')
# declare -a splits=('length' 'no_mod' 'random_cvcv' 'random_str')
declare -a splits=('simple' 'length' 'addprim_jump' 'addprim_turn_left' 'jump_random_cvcv' 'jump_random_str' 'turn_left_random_cvcv' 'turn_left_random_str' 'mcd1' 'mcd2' 'mcd3' 'template_around_right')

cd $BASE_DIR

if [[ $dataset == *"hp"* ]]
then
    python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/train.tsv" --output_source="data/${dataset}/btg_source_train.txt" --output_target="data/${dataset}/btg_target_train.txt"

    python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/test.tsv" --output_source="data/${dataset}/btg_source_test.txt" --output_target="data/${dataset}/btg_target_test.txt"
else
    for split in "${splits[@]}"
    do

        python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/${split}_split/train.tsv" --output_source="data/${dataset}/${split}_split/btg_source_train.txt" --output_target="data/${dataset}/${split}_split/btg_target_train.txt"

        python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/${split}_split/test.tsv" --output_source="data/${dataset}/${split}_split/btg_source_test.txt" --output_target="data/${dataset}/${split}_split/btg_target_test.txt"

        if [[ $dataset == *"COGS"* ]]
        then
            python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/${split}_split/gen.tsv" --output_source="data/${dataset}/${split}_split/btg_source_gen.txt" --output_target="data/${dataset}/${split}_split/btg_target_gen.txt"

            python utils/helper_utils/strip_source_and_target.py --input="data/${dataset}/${split}_split/dev.tsv" --output_source="data/${dataset}/${split}_split/btg_source_dev.txt" --output_target="data/${dataset}/${split}_split/btg_target_dev.txt"
        fi

    done
fi



