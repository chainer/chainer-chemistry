#!/usr/bin/env bash

set -e

# gpu id given from first argument, default value is -1
gpu=${1:--1}

for method in nfp ggnn schnet weavenet rsgcn
do
    # QM9
    if [ ! -f "input" ]; then
        rm -rf input
    fi

    python train_qm9.py --method ${method} --label A --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32 --num-data 100
    python predict_qm9.py --method ${method} --label A --gpu ${gpu} --batchsize 32 --num-data 100
    python train_qm9.py --method ${method} --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32 --num-data 100
    python predict_qm9.py --method ${method} --gpu ${gpu} --batchsize 32 --num-data 100
done
