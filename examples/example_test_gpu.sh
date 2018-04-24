#!/usr/bin/env bash

set -e

gpu=0

declare method_list=(nfp ggnn schnet weavenet rsgcn)
# Preprocessor parse result must contain both pos/neg samples
declare tox21_num_data_list=(1000 1000 2000 3000 1000)

#for method in nfp ggnn schnet weavenet rsgcn
for (( i=0; i<${#method_list[@]}; i++ ));
do
    method=${method_list[$i]}
    tox21_num_data=${tox21_num_data_list[$i]}

    # Tox21
    cd tox21
    if [ ! -f "input" ]; then
        rm -rf input
    fi

    # Tox21 classification task with only one label
    out_dir=nr_ar_${method}
    python train_tox21.py --method ${method} --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out ${out_dir} --batchsize 32 --num-data=${tox21_num_data}
    python inference_tox21.py --in-dir ${out_dir} --gpu ${gpu} --num-data=${tox21_num_data}
    python predict_tox21_with_classifier.py --in-dir ${out_dir} --gpu ${gpu} --num-data=${tox21_num_data}
    snapshot=`ls ${out_dir}/snapshot_iter_* | head -1`
    python inference_tox21.py --in-dir ${out_dir} --gpu ${gpu} --trainer-snapshot ${snapshot} --num-data=${tox21_num_data}

    # Tox21 classification task with all labels
    out_dir=all_${method}
    python train_tox21.py --method ${method} --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out ${out_dir} --batchsize 16 --num-data=${tox21_num_data}
    python inference_tox21.py --in-dir ${out_dir} --num-data=${tox21_num_data}
    python predict_tox21_with_classifier.py --in-dir ${out_dir} --num-data=${tox21_num_data}
    snapshot=`ls ${out_dir}/snapshot_iter_* | head -1`
    python inference_tox21.py --in-dir ${out_dir} --gpu ${gpu} --trainer-snapshot ${snapshot} --num-data=${tox21_num_data}
    cd ../

    # QM9
    cd qm9
    if [ ! -f "input" ]; then
        rm -rf input
    fi

    python train_qm9.py --method ${method} --label A --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32 --num-data 100
    python predict_qm9.py --method ${method} --label A --gpu ${gpu} --batchsize 32 --num-data 100
    python train_qm9.py --method ${method} --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32 --num-data 100
    python predict_qm9.py --method ${method} --gpu ${gpu} --batchsize 32 --num-data 100
    cd ../

    # Own dataset
    cd own_dataset
    python train.py dataset.csv --method ${method} --label value1 --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --batchsize 32
    cd ../
done

cd tox21
# BalancedSerialIterator test with Tox21
python train_tox21.py --method nfp --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out nr_ar_nfp_balanced --iterator-type balanced --eval-mode 0 --num-data 1000
python inference_tox21.py --in-dir nr_ar_nfp_balanced --gpu ${gpu} --num-data 1000
# ROCAUCEvaluator test with Tox21
python train_tox21.py --method nfp --label NR-AR --conv-layers 1 --gpu ${gpu} --epoch 1 --unit-num 10 --out nr_ar_nfp_balanced --iterator-type serial --eval-mode 1 --num-data 1000
cd ..
