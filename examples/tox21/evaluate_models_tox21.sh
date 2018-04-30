set -eu

gpu=-1
methods=(nfp ggnn schnet weavenet rsgcn)
results=()

mkdir -p results-evaluation
for method in ${methods[@]}
do
    python train_tox21.py --method ${method} --gpu ${gpu}
    python predict_tox21_with_classifier.py > log-${method}
    roc_auc=`cat log-${method} | tail -1 | grep -oP '0\.\d+'`
    results+=(${roc_auc})
    mv result results-evaluation/${method}
done

python plot.py --names ${methods[@]} --values ${results[@]}
