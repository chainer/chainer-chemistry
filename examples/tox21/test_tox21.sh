#!/usr/bin/env bash

set -e

# gpu id given from first argument, default value is -1
gpu=${1:--1}

# Preprocessor parse result must contain both pos/neg samples
tox21_num_data=100

for method in nfp ggnn schnet weavenet rsgcn
do
    if [ ! -f "input" ]; then
        rm -rf input
    fi

    # Tox21 classification task with only one label
    out_dir=nr_ar_${method}
    python train_tox21.py --method ${method} --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out ${out_dir} --batchsize 32 --num-data=${tox21_num_data}
    python predict_tox21_with_classifier.py --in-dir ${out_dir} --gpu ${gpu} --num-data=${tox21_num_data}

    # Tox21 classification task with all labels
    out_dir=all_${method}
    python train_tox21.py --method ${method} --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out ${out_dir} --batchsize 16 --num-data=${tox21_num_data}
    python predict_tox21_with_classifier.py --in-dir ${out_dir} --num-data=${tox21_num_data}
done

# BalancedSerialIterator test with Tox21
python train_tox21.py --method nfp --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out nr_ar_nfp_balanced --iterator-type balanced --eval-mode 0 --num-data 1000
# ROCAUCEvaluator test with Tox21
python train_tox21.py --method nfp --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out nr_ar_nfp_balanced --iterator-type serial --eval-mode 1 --num-data 1000
