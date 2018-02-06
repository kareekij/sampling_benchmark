#!/bin/bash

search_dir="data-syn-mixing"
echo $search_dir

for path in "$search_dir"/*; do
    [ -d "${path}" ] || continue # if not a directory, skip
    target_file="$(basename "${path}")"
    echo $target_file
    python ./calculate_syn_graph.py $target_file
done

# for file in "$search_dir"/*
# do
#   if [ ! -d "$file" ]; then
#       target_file="./$file"
#       echo $target_file
#       python ./calculate_properties_from_sample_file.py $target_file
#   fi
# done