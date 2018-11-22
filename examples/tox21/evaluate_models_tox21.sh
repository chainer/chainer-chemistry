set -eu

gpu=-1
methods=(nfp ggnn schnet weavenet rsgcn)
prefix=eval_

for method in ${methods[@]}
do
    result_dir=${prefix}${method}
    python train_tox21.py --method ${method} --gpu ${gpu} --out ${result_dir}
    python predict_tox21_with_classifier.py --in-dir ${result_dir}
done

python plot.py --prefix ${prefix} --methods ${methods[@]}
