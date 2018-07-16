#!/usr/bin/env bash

set -e

declare dataset_list=(bace_Class bace_pIC50 bbbp clearance clintox delaney HIV hopv lipo muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)

# gpu id given from first argument, default value is -1
gpu=${1:--1}

for (( i=0; i<${#dataset_list[@]}; i++ ));
do

    if [ ! -f "input" ]; then
        rm -rf input
    fi

    dataset=${dataset_list[$i]}
    out_dir=nfp_${dataset}
    echo $dataset
    python train_molnet.py --dataset $dataset --method nfp --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out ${out_dir} --batchsize 32 --num-data=100
done
