#!/bin/bash

search_dir="./data-fb"
echo $search_dir

for file in "${search_dir}/"*.* ; do
    #target_file="$(basename "${file}")"
    #filename=$(echo $target_file| cut -d'_' -f 1)
    echo "Executing.. "$file
    python calculate_properties_step_real.py $file
done
