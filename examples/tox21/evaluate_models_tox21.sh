set -eu

gpu=-1
methods=(nfp ggnn schnet weavenet rsgcn)
dataset="tox21"
output_file="eval_results_${dataset}.png"

for method in ${methods[@]}
do
    result_dir="eval_${method}"
    python train_tox21.py --method ${method} --gpu ${gpu} --out ${result_dir}
    python predict_tox21_with_classifier.py --in-dir ${result_dir}
done

python ../utils/plot_roc.py --methods ${methods[@]} --dataset ${dataset} --output-file ${output_file}
