#!/bin/bash -eu

run=$1
nle=$2

device=0
dataset_list=(bace_Class bace_pIC50 bbbp clearance clintox delaney HIV hopv lipo muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)
#methods=(rsgcn)
methods=(rsgcn relgcn relgat ggnn)
epoch=500
prefix=eval_test
#prefix=eval
runs=10
batchsize=128
nlayers=8
#unit_num=100
unit_nums=(100 40 40 72)

dataset=qm9

if [ $nle = "1" ] ; then
    prefix=${prefix}_NLE
fi

for nlayer in `seq 2 $nlayers`
do
    for m in `seq 0 0`
    #for method in ${methods[@]}
    do
	method=${methods[${m}]}
	unit_num=${unit_nums[${m}]}
	echo ""
	echo "nlayer=$nlayer "
	echo "method=$method "
	echo "unit_num=$unit_num "
	echo "run=$run "
	echo "nle=$nle "
	
	result_dir=${prefix}_layer${nlayer}_${dataset}_${method}_${run}
	
	if [ $nle = "0" ] ; then
	    echo "NLE disabled"
	    
	    #echo "PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir}"
	    #PYTHONPATH=/home/ishiguro/Project/nn_theory/191003_NeighborLabelExpansion/chainer-chemistry-v6 python examples/molnet/train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir}
	    PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir}  --scale none

	elif [ $nle = "1" ] ; then
	    echo "NLE enabled"

	    #echo "PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir} --apply_nle"

	    #PYTHONPATH=/home/ishiguro/Project/nn_theory/191003_NeighborLabelExpansion/chainer-chemistry-v6 python examples/molnet/train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir} --apply-nle  --num-data 500
	    PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/train_molnet.py --dataset ${dataset} --method ${method} --conv-layers ${nlayer} --device ${device} --epoch ${epoch} --batchsize ${batchsize} --unit-num ${unit_num} --out ${result_dir} --scale none --apply-nle
	else
	    echo "fishy nle"
	fi

	
	#echo "PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python predict_molnet.py --device ${device} --dataset ${dataset} --method ${method} --in-dir ${result_dir}"
	#PYTHONPATH=/home/ishiguro/Project/nn_theory/191003_NeighborLabelExpansion/chainer-chemistry-v6 python examples/molnet/predict_molnet.py --device ${device} --dataset ${dataset} --method ${method} --in-dir ${result_dir} --num-data 100
	PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/predict_molnet.py --device ${device} --dataset ${dataset} --method ${method} --in-dir ${result_dir}
	
	cp -r ${result_dir} /mnt/vol12/ishiguro/temp/${result_dir}
    done # end method-for

    #result_dir=${prefix}_${dataset}_${method}_${nlayer}
    #PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python example/molent/summary_eval_molnet.py --prefix /mnt/vol12/ishiguro/temp/${result_dir} --methods ${methods[@]} --dataset ${dataset} --runs ${runs} --out_prefix /mnt/vol12/ishiguro/temp_eval/${result_dir}/${prefix}_{$
    
done # end layer-for


    
