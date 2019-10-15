#!/bin/bash -eu

dataset_list=(bace_Class bace_pIC50 bbbp clearance clintox delaney HIV hopv lipo muv nci pcba ppb qm7 qm8 qm9 SAMPL sider tox21 toxcast)
methods=(rsgcn)
#methods=(rsgcn relgat ggnn)
#epoch=500
in_dir=/mnt/vol12/ishiguro/Project/nn-theory/191003_NeighborLabelExpansion/191014_feasibilityExp/
temp_dir=/mnt/vo12/ishiguro/temp_temp
out_dir=/mnt/vol12/ishiguro/temp/
prefix=eval_test
runs=10

#batchsize=128
nlayers=8
#unit_num=100

datasets=(qm9 tox21 HIV lipo)


for dataset in ${datasets[@]}
do
    
    for nlayer in `seq 2 $nlayers`
    do
	
	# run the summary script
	indir_prefix=${in_dir}${prefix}_layer${nlayer}
	out_prefix=${out_dir}${prefix}_layer${nlayer}_${dataset}/result
	
	echo ""
	echo "indir_prefix=$indir_prefix "
	echo "out_prefix=$out_prefix "
	
	PYTHONPATH=/home/ishiguro/workspace/chainer-chemistry-v6 python examples/molnet/summary_eval_molnet.py --indir_prefix ${indir_prefix} --methods ${methods[@]} --dataset ${dataset} --runs ${runs} --out_prefix ${out_prefix}
	
	
    done # end layer-for
done # end method-for


    
