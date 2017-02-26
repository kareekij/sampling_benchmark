#!/bin/bash

gen_model_count="$1"
name="$2"

if [ "$#" -ne 2 ]; then
    echo "Incorrent params"
    echo "./run.sh <number_of_gen_model> <name>"
    exit 1
fi

echo "Generate" $gen_model_count "models"
echo "Name:" $name
counter=1

while [ $counter -le $gen_model_count ]
do
  echo "Generating LFR Model" $counter

  DIRECTORY="gen/$name/$counter"

  if [ ! -d "$DIRECTORY" ];then
    echo $DIRECTORY
    mkdir -p $DIRECTORY
  fi

  ./benchmark
  mv network.dat $DIRECTORY
  mv community.dat $DIRECTORY
  mv statistics.dat $DIRECTORY

  ((counter++))
done

#./benchmark
