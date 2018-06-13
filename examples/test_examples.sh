#!/usr/bin/env bash

set -e

gpu=${1:--1}
echo Using gpu ${gpu}

# Tox21
echo --- Testing Tox21 ---
cd tox21 && bash -x test_tox21.sh ${gpu} && cd ..

# QM9
echo --- Testing QM9 ---
cd qm9 && bash -x test_qm9.sh ${gpu} && cd ..

# Own dataset
echo --- Testing Own dataset ---
cd own_dataset && bash -x test_own_dataset.sh ${gpu} && cd ..

# MolNet
echo --- Testing MolNet dataset ---
cd molnet && bash -x test_molnet.sh ${gpu} && cd ..
