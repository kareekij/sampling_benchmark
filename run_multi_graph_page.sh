#!/bin/bash

dataset="$1"
budget="$2"
page="$3"

if [ "$#" -ne 3 ]; then
    echo "Incorrent params"
    echo "./run.sh <dataset> <budget> <is_page>"
    exit 1
fi
#
#echo "Generate" $gen_model_count "models"
#echo "Name:" $name
counter=1
#
while [ $counter -le 10 ]
do

  file_name="./data-syn/${dataset}/$counter/network.dat"
  #file_name="./data/${dataset}/$counter/network.dat"
  #file_name="./data/${dataset}/$counter/network.dat"
  echo "Running $file_name"
  echo sample_incomp.py $file_name -dataset $dataset -budget $budget -experiment 1 -is_Page $page
  python sample_incomp.py $file_name -dataset $dataset -budget $budget -experiment 1 -is_Page $page

  ((counter++))
done

#./benchmark
