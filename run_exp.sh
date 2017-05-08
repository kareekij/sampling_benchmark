#!/usr/bin/env bash

search_dir="$1"

if [ "$#" -ne 1 ]; then
    echo "Incorrect params"
    echo "./run_exp.sh <folder_name>"
    exit 1
fi

for entry in "$search_dir"/*
do
  echo "Running $entry"
  python sample.py $entry

done