set -eu

gpu=-1
methods=(nfp ggnn schnet weavenet rsgcn)
dir_name="result-evaluation"
results=()

mkdir -p ${dir_name}
for method in ${methods[@]}
do
    python train_tox21.py --method ${method} --gpu ${gpu} -c 1 -u 1 -e 1
    python predict_tox21_with_classifier.py
    mv result ${dir_name}/${method}
    mv result.json ${dir_name}/${method}.json
done

python plot.py --dir ${dir_name} --methods ${methods[@]}
