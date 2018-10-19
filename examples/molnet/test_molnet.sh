#!/usr/bin/env bash

set -e

# List of available datasets.
# TODO: Investigate why training on `clearance` fails.
datasets=(bace_Class bace_pIC50 bbbp clintox delaney HIV hopv lipo \
          muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)

# GPU identifier; set it to -1 to train on the CPU (default).
gpu=${1:--1}

# Remove directories with previously trained models.
[ -d input ] && rm -rf input

for dataset in ${datasets[@]}
do
    # Run the training script for the current dataset.
    python train_molnet.py \
        --dataset $dataset \
        --method nfp \
        --conv-layers 1 \
        --gpu ${gpu} \
        --epoch 1 \
        --unit-num 10 \
        --out nfp_${dataset} \
        --batchsize 32 \
        --num-data=100
done
