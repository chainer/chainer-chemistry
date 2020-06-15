#!/bin/bash -eu

method_idx=$1
nle=$2
k=$3
device=$4

dataset_list=(bace_Class bace_pIC50 bbbp clearance clintox delaney HIV hopv lipo muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)

methods=(rsgcn relgcn relgat ggnn)

epoch=250
#epoch=2
trial=100
#trial=2

prefix=eval_test

batchsize=128

#dataset=qm9
dataset=qm8

method=${methods[${method_idx}]}

datetime=`date +"%y%m%d_%H%M%S"`
out_prefix=log/$datetime/${prefix}_${dataset}


if [ -d input ]; then
    rm -rf input
fi


if  [ ${nle} = "0" ] ; then
    echo "NLE disabled"
    method_str=${method}
    
    PYTHONPATH=. python examples/molnet/optuna_train_molnet.py --dataset ${dataset} --method ${method_str} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --out ${out_prefix}_${method_str} --scale none --num-trial ${trial} --num-hop ${k}
    
elif [ ${nle} = "1" ] ; then
    echo "NLE enabled"
    method_str=${method}
    
    PYTHONPATH=. python examples/molnet/optuna_train_molnet.py --dataset ${dataset} --method ${method_str} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --out ${out_prefix}_${method_str}_nle --scale none --num-trial ${trial} --num-hop ${k} --apply-nle 

elif [ ${nle} = "2" ] ; then
    echo "CNLE enabled"
    method_str=${method}_cnle
    
    PYTHONPATH=. python examples/molnet/optuna_train_molnet.py --dataset ${dataset} --method ${method_str} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --out ${out_prefix}_${method_str} --scale none --num-trial ${trial} --num-hop ${k} --apply-cnle

elif [ ${nle} = "3" ] ; then
    echo "GCNLE enabled"
    method_str=${method}_gcnle

    PYTHONPATH=. python examples/molnet/optuna_train_molnet.py --dataset ${dataset} --method ${method_str} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --out ${out_prefix}_${method_str} --scale none --num-trial ${trial} --num-hop ${k} --apply-gcnle

else
    echo "fishy nle"
fi
