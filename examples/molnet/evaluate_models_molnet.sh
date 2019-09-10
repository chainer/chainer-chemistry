#!/usr/bin/env bash
set -e

# List of available datasets.
# TODO: Investigate why training on `clearance` fails.
datasets=(bace_Class bace_pIC50 bbbp clintox delaney HIV hopv lipo \
          muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)
methods=(relgcn)

# device identifier; set it to -1 to train on the CPU (default).
device=${1:--1}

# Remove directories with previously trained models.
[ -d result ] && rm -rf result

for dataset in ${datasets[@]}; do
    for method in ${methods[@]}; do
        python train_molnet.py \
            --dataset ${dataset} \
            --method ${method} \
            --device ${device} \
            --epoch 1 \
            --unit-num 10 \
            --conv-layers 1 \
            --num-data 100 \
            --out result

        python predict_molnet.py \
            --dataset ${dataset} \
            --method ${method} \
            --in-dir result \
            --device ${device} \
            --num-data 100
    done
done
