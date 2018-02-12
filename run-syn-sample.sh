#!/bin/bash

search_dir="./data-syn/multi-mixing-0.2"
new_name="mixing02"
echo $search_dir

for file in "${search_dir}/"*; do
    target_file="$(basename "${file}")"
    folder_number=$(echo $target_file| cut -d'_' -f 1)
    new_path=$search_dir"/"$new_name"_"$folder_number".txt"
    cp $file"/network.dat" $new_path
    echo "Copying.. "$new_path

    #python calculate_properties_step_syn.py $filename
done
