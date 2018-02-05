#!/bin/bash

search_dir="data-control-real"
echo $search_dir
for file in "$search_dir"/*
do
  if [ ! -d "$file" ]; then
      target_file="./$file"
      echo $target_file
      python ./calculate_properties_from_sample_file.py $target_file
      #python ./calculate_syn_graph.py $target_file
  fi
done