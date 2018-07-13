#!/usr/bin/env bash

set -e

# gpu id given from first argument, default value is -1
gpu=${1:--1}

for method in nfp ggnn schnet weavenet rsgcn
do
    python train.py --datafile dataset.csv --method ${method} --label value1 --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32
done
