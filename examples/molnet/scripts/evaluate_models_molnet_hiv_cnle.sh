#!/bin/bash -eu

run=$1

device=0
dataset_list=(bace_Class bace_pIC50 bbbp clearance clintox delaney HIV hopv lipo muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)
methods=(rsgcn_cnle relgat_cnle ggnn_cnle)
epoch=500
prefix=eval_test
runs=10
batchsizes=(128 32 128)
nlayers=(2 3 2)
unit_nums=(45 20 45)
adam_alphas=(0.0011 0.000065 0.0017)

dataset=HIV

prefix=${prefix}_CNLE


for m in `seq 0 2`
do
    batchsize=${batchsizes[${m}]}
    method=${methods[${m}]}
    unit_num=${unit_nums[${m}]}
    nlayer=${nlayers[${m}]}
    adam_alpha=${adam_alphas[${m}]}
    echo ""
    echo "nlayer=$nlayer "
    echo "method=$method "
    echo "unit_num=$unit_num "
    echo "adam_alpha=$adam_alpha "
    echo "run=$run "
		
    result_dir=${prefix}_layer${nlayer}_${dataset}_${method}_${run}
    
    PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir} --adam-alpha ${adam_alpha} --apply-cnle
	
    PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/predict_molnet.py --device ${device} --dataset ${dataset} --method ${method} --in-dir ${result_dir}
    
    #cp -r ${result_dir} /mnt/vol12/ishiguro/temp_hiv/${result_dir}

done
