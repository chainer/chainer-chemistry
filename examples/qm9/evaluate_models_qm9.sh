set -eu

gpu=-1
methods=(nfp ggnn schnet weavenet rsgcn)
prefix=eval_

for method in ${methods[@]}
do
    result_dir=${prefix}${method}
    python train_qm9.py --method ${method} --gpu ${gpu} --out ${result_dir} --epoch 1
    python predict_qm9.py --in-dir ${result_dir} --method ${method}
done

python plot.py --prefix ${prefix} --methods ${methods[@]}
