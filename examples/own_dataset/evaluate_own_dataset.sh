#!/usr/bin/env bash

set -e

# List of available graph convolution methods.
methods=(nfp ggnn schnet weavenet rsgcn)
# GPU identifier; set it to -1 to train on the CPU (default).
gpu=${1:--1}
# Number of training epochs (default: 1).
epoch=${2:-1}

for method in ${methods[@]}
do
    # Train with the current method.
    python train_own_dataset.py \
        --method ${method} \
        --label value1 \
        --conv-layers 1 \
        --gpu ${gpu} \
        --epoch ${epoch} \
        --unit-num 10 \
        --out eval_${method}

    # Run inference on the test set.
    python predict_own_dataset.py \
        --method ${method} \
        --label value1 \
        --conv-layers 1 \
        --gpu ${gpu} \
        --epoch ${epoch} \
        --unit-num 10 \
        --in-dir eval_${method} \
        --out eval_${method}
done

# Create plot showing the evaluation performance.
python plot.py --prefix eval_ --methods ${methods[@]}
