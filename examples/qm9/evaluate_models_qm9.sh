set -eu

# List of available graph convolution methods.
methods=(nfp ggnn schnet weavenet rsgcn relgcn)

prefix=eval_

# GPU identifier; set it to -1 to train on the CPU (default).
gpu=${1:--1}
# Number of training epochs (default: 1).
epoch=${2:-1}

for method in ${methods[@]}
do
    result_dir=${prefix}${method}

    python train_qm9.py \
        --method ${method} \
        --gpu ${gpu} \
        --out ${result_dir} \
        --epoch ${epoch}

    python predict_qm9.py \
        --in-dir ${result_dir} \
        --method ${method}
done

python plot.py --prefix ${prefix} --methods ${methods[@]}
