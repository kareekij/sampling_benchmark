#!/bin/bash

#search_dir="data-control-real"
search_dir=$1
echo $search_dir
for file in "$search_dir"/*
do
  if [ ! -d "$file" ]; then
      target_file="./$file"
      echo $target_file
      python sample.py $target_file -mode 3
  fi
done
