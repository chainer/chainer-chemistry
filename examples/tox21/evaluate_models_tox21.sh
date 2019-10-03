set -eu

device=-1
methods=(nfp ggnn schnet weavenet rsgcn relgcn relgat megnet)
prefix=eval_

for method in ${methods[@]}
do
    result_dir=${prefix}${method}
    python train_tox21.py --method ${method} --device ${device} --out ${result_dir}
    python predict_tox21_with_classifier.py --in-dir ${result_dir}
done

python plot.py --prefix ${prefix} --methods ${methods[@]}
