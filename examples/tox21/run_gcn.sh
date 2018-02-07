#!/bin/bash -x

# export CHAINER_USE_CUDNN=never

HOST=`hostname -s`
DATE=`date +%y%m%d-%H%M%S`

#
# gcn
#
METHOD=gcn
# METHOD=gcn_opt1
BATCH=128
EPOCH=20
UNIT=32
LAYER=4

NAME=m_$METHOD.b_$BATCH.e_$EPOCH.u_$UNIT.l_$LAYER.$HOST.$DATE

mkdir logs 2> /dev/null
LOG_OUT=logs/out.$NAME
LOG_NVPROF=logs/nvprof.$NAME

# nvprof -o $LOG_NVPROF \
python train_tox21.py \
	  --method $METHOD \
	  --batchsize $BATCH \
	  --epoch $EPOCH \
	  --unit-num $UNIT \
	  --conv-layers $LAYER \
	  --gpu 1 \
    | tee $LOG_OUT
