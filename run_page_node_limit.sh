#!/usr/bin/env bash

dataset="$1"
isPage="$2"
limit="$3"

echo$limit

if [ "$#" -ne 3 ]; then
    echo "Incorrent params"
    echo "./run.sh <dataset> <isPage> <node_lim>"
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
  #file_name="./data-syn-06/${dataset}/$counter/network.dat"

  echo "Running $file_name"
  echo sample_incomp.py $file_name -dataset $dataset -budget 1000 -experiment 1 -mode 1 -is_Page $isPage -node_limit $limit
  python sample_incomp.py $file_name -dataset $dataset -budget 1000 -experiment 1 -mode 1 -is_Page $isPage -node_limit $limit

  ((counter++))
done

#./benchmark
