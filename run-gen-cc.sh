#!/bin/bash

exp_cc="$1"
gen_model_count="$2"

if [ "$#" -ne 2 ]; then
    echo "Incorrent params"
    echo "./run.sh <expected_cc>  <gen_count>"
    exit 1
fi

echo "CC:"$exp_cc
counter=1

while [ $counter -le $gen_model_count ]
do
  echo "Generating Model with Desired CC" $counter

  DIRECTORY="./data/cc-$exp_cc/$counter"

  if [ ! -d "$DIRECTORY" ];then
    echo $DIRECTORY
    mkdir -p $DIRECTORY
  fi

  python ./adjust_cc.py $exp_cc
  mv network.dat $DIRECTORY

  ((counter++))
done

#./benchmark
#!/usr/bin/env bash