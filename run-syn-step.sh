#!/bin/bash

search_dir="/Users/Katchaguy/Google Drive/results/imc2017/log-syn-mix-low"
echo $search_dir

for file in "${search_dir}/"*_order.txt ; do
    target_file="$(basename "${file}")"
    filename=$(echo $target_file| cut -d'_' -f 1)
    echo "Executing.. "$filename
    python calculate_properties_step_syn.py $filename
done
