#!/usr/bin/env bash

set -e

# List of available graph convolution methods.
methods=(megnet)

# device identifier; set it to -1 to train on the CPU (default).
device=${1:--1}
# Number of training epochs (default: 1).
epoch=${2:-1}

for method in ${methods[@]}
do
    # Remove any previously cached models.
    [ -d "input" ] && rm -rf input

    # Train with the current method (one label).
    python train_mp.py \
        --method ${method} \
        --label band_gap \
        --conv-layers 1 \
        --device ${device} \
        --epoch ${epoch} \
        --unit-num 10 \
        --num-data 100

    # Predict with the current method (one label).
    python predict_mp.py \
        --method ${method} \
        --label band_gap \
        --device ${device} \
        --num-data 100

done
