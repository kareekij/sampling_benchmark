#!/usr/bin/env bash

dataset="$1"
page="$2"

if [ "$#" -ne 2 ]; then
    echo "Incorrent params"
    echo "./run.sh <dataset> <is_page>"
    exit 1
fi

counter=1

while [ $counter -le 10 ]
do

  file_name="./data-syn/${dataset}/$counter/network.dat"
  #file_name="./data-syn-06/${dataset}/$counter/network.dat"

  echo "Running $file_name"
  echo sample_incomp.py $file_name -dataset $dataset -budget 1000 -mode 1 -experiment 1 -is_Page $page
  python sample_incomp.py $file_name -dataset $dataset -budget 1000 -mode 1 -experiment 1 -is_Page $page

  ((counter++))
done

