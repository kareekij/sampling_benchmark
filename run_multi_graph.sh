#!/bin/bash

dataset="$1"
budget="$2"

if [ "$#" -ne 2 ]; then
    echo "Incorrent params"
    echo "./run.sh <dataset> <budget>"
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
  #python sample.py $file_name -dataset $dataset -budget $budget -experiment 1
  python sample_page.py $file_name -dataset $dataset -budget $budget -experiment 1

  ((counter++))
done

#./benchmark
