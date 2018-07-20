#!/usr/bin/env bash

set -e
methods=(nfp ggnn schnet weavenet rsgcn)

# gpu id given from first argument, default value is -1
gpu=${1:--1}

for method in ${methods[@]}
do
    python train_custom_dataset.py \
        --method ${method} \
        --label value1 \
        --conv-layers 1 \
        --gpu ${gpu} \
        --epoch 1 \
        --unit-num 10 \
        --out eval_${method}

    python predict_custom_dataset.py \
        --method ${method} \
        --label value1 \
        --conv-layers 1 \
        --gpu ${gpu} \
        --epoch 1 \
        --unit-num 10 \
        --in-dir eval_${method} \
        --out eval_${method}
done

python plot.py --prefix ${prefix} --methods ${methods[@]}
