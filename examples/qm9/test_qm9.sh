#!/usr/bin/env bash

set -e

# List of available graph convolution methods.
methods=(nfp ggnn schnet weavenet rsgcn)
# Number of training epochs (default: 1).
epoch=${1:-1}
# GPU identifier; set it to -1 to train on the CPU (default).
gpu=${2:--1}

for method in ${methods[@]}
do
	# Remove any previously cached models.
	[ -d "input" ] && rm -rf input

	# Train with the current method (one label).
	python train_qm9.py \
		--method ${method} \
		--label A \
		--conv-layers 1 \
		--gpu ${gpu} \
		--epoch ${epoch} \
		--unit-num 10 \
		--num-data 100

	# Predict with the current method (one label).
	python predict_qm9.py \
		--method ${method} \
		--label A \
		--gpu ${gpu} \
		--num-data 100

	# Train with the current method (all labels).
	python train_qm9.py \
		--method ${method} \
		--conv-layers 1 \
		--gpu ${gpu} \
		--epoch ${epoch} \
		--unit-num 10 \
		--num-data 100

	# Predict with the current method (all labels).
	python predict_qm9.py \
		--method ${method} \
		--gpu ${gpu} \
		--num-data 100
done
