#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os

import chainer
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator  # NOQA
from chainer import cuda
# Proposed by Ishiguro
# ToDo: consider go/no-go with following modification
# Re-load the best-validation score snapshot using serializers
from chainer import serializers

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.utils import save_json

# These import is necessary for pickle to work
from chainer import functions as F
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler  # NOQA
from chainer_chemistry.models.prediction.graph_conv_predictor import GraphConvPredictor  # NOQA
from train_molnet import dataset_part_filename
from train_molnet import download_entire_dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'gin', 'gnnfilm',
                   'nfp_cnle', 'ggnn_cnle', 'relgat_cnle', 'relgcn_cnle', 'rsgcn_cnle', 'gin_cnle',
                   'nfp_gcnle', 'ggnn_gcnle', 'relgat_gcnle', 'relgcn_gcnle', 'rsgcn_gcnle', 'gin_gcnle']
#    scale_list = ['standardize', 'none']
    dataset_names = list(molnet_default_config.keys())

    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Prediction on Molnet.')
    parser.add_argument('--dataset', '-d', type=str, choices=dataset_names, required=True, help='name of the dataset that training is run on')
    parser.add_argument('--method', '-m', type=str, choices=method_list, required=True, help='method name')
    parser.add_argument('--wl-list', '-w', type=str, required=True, help='list file of WL labels,Frequency')
    parser.add_argument('--label', '-l', type=str, default='',
                        help='target label for regression; empty string means '
                        'predicting all properties at once')
#    parser.add_argument('--scale', type=str, choices=scale_list,
#                        help='label scaling method', default='standardize')
    parser.add_argument(
        '--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory to load model data from')

    parser.add_argument('--out-prefix', '-o', type=str, default="result", help="output prefix to dump files")
    parser.add_argument('--num-data', type=int, default=1000,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')
    parser.add_argument('--batchsize', '-b', type=int, default=250,
                        help='batch size')

    return parser.parse_args()


def main():
    """
    Load the dataset and the learned model.
    Then hack into the parameters / variables for CNLE and GCNLE.


    :return:
    """
    args = parse_arguments()

    cnle_methods = ['nfp_cnle', 'ggnn_cnle', 'relgat_cnle', 'relgcn_cnle', 'rsgcn_cnle', 'gin_cnle',]
    gcnle_methods = ['nfp_gcnle', 'ggnn_gcnle', 'relgat_gcnle', 'relgcn_gcnle', 'rsgcn_gcnle', 'gin_gcnle',]

    # Set up some useful variables that will be used later on.
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data
    out_prefix = args.out_prefix
    outprefix_lastslash = args.out_prefix.rindex("/")
    if outprefix_lastslash > -1:
        outdir = args.out_prefix[:outprefix_lastslash]

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    apply_nle_flag = False
    apply_cnle_flag = False
    apply_gcnle_flag = False

    if method in cnle_methods:
        apply_cnle_flag = True
    elif method in gcnle_methods:
        apply_gcnle_flag = True
    else:
        raise ValueError("invalid method type")
    print("nle_flag=", apply_nle_flag)
    print("cnle_flag=",  apply_cnle_flag)
    print("tcnle_flag=",  apply_gcnle_flag)

    #
    # load the list of WL labels
    #

    with open(args.wl_list, "r") as fin:
        wl_list_lines = fin.readlines()
    num_WLs = len(wl_list_lines)
    print("Number of WLs=", num_WLs)
    print(wl_list_lines[0])
    print(wl_list_lines[1])
    WL_list = []
    WL_freq = []
    for line in wl_list_lines:
        wl, freq = line.rstrip().split(" ")
        WL_list.append(wl)
        WL_freq.append(int(freq))
    print(WL_list[0], WL_freq[0])
    print(WL_list[1], WL_freq[1])

    if args.label:
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}_{}'.format(dataset_name,
                                                            method, labels))
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_{}_all'.format(dataset_name,
                                                             method))

    # Load the cached dataset.
    filename = dataset_part_filename('test', num_data)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        print('Loading cached dataset from {}.'.format(path))
        test = NumpyTupleDataset.load(path)
    else:
        _, _, test = download_entire_dataset(dataset_name, num_data, labels,
                                             method, cache_dir, apply_nle_flag, 0, apply_cnle_flag, apply_gcnle_flag)

    # Model-related data is stored this directory.
    model_dir = os.path.join(args.in_dir, os.path.basename(cache_dir))

    model_filename = {'classification': 'classifier.pkl',
                      'regression': 'regressor.pkl'}
    task_type = molnet_default_config[dataset_name]['task_type']
    model_path = os.path.join(model_dir, model_filename[task_type])
    print("model_path=" + model_path)
    print('Loading model weights from {}...'.format(model_path))

    device = chainer.get_device(args.device)
    if task_type == 'classification':
        model = Classifier.load_pickle(model_path, device=device)
    elif task_type == 'regression':
        model = Regressor.load_pickle(model_path, device=device)
    else:
        raise ValueError('Invalid task type ({}) encountered when processing '
                         'dataset ({}).'.format(task_type, dataset_name))

    # check visual
    test_iterator = SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    
    import numpy as np
    data = test_iterator.__next__() # is a minibatch-length list
    print("len(data)=", len(data))
    
    if apply_cnle_flag:
        #print(model)
        #print(model.predictor)
        cnle = model.predictor.graph_conv
        print(cnle)
        #concatW = cwle.linear_for_concat_nle.W.data
        #print(type(concatW))

        embed = cnle.embed
        embed_nle = cnle.embed_nle
        #print(embed)
        #print(embed_nle)

        embed_W = embed.W.data
        embed_nle_W = embed_nle.W.data
        print(embed_W.shape)
        print(embed_nle_W.shape)
        # dump the embedding

        out_prefix = out_prefix + "_WL_Embed"
        with open(out_prefix + ".dat", 'w') as fout:
            import csv
            writer = csv.writer(fout, lineterminator="\n")
            writer.writerows(embed_nle_W[:num_WLs, :])
        # end with

        import pickle
        with open(out_prefix + "_WL_Embed.pkl", "wb") as fout:
            pickle.dump(embed_nle_W[:num_WLs, :], fout)
        # end-with

    elif apply_gcnle_flag:
        gcnle = model.predictor.graph_conv
        print(gcnle)
        
        shape_W = gcnle.gate_W1.W.shape
        print(shape_W)
        gate_coefficients = np.zeros( (len(data), shape_W[0] ) )
        for i, data_i in enumerate(data):
            print( len(data_i))
            atom_array = data_i[0]
            adj = data_i[1]
            nle_array = data_i[2]
            print(atom_array)
            print(adj)
            print(nle_array)
            #target_y = data_i[3]

            # forward manually
            gcnle.reset_state()

            h = gcnle.embed(atom_array).data
            h_s = gcnle.embed_nle(nle_array).data
            h2 = (np.expand_dims(h, axis=0))
            h2_s = (np.expand_dims(h_s, axis=0))
            gate_input = gcnle.gate_W1(h2) + gcnle.gate_W2(h2_s)
            gate_coefff = chainer.functions.sigmoid(gate_input)
            print(gate_coefff.shape)

            gate_coefficients[i] = np.mean(np.mean(gate_coefff))
        # end for
        print(gate_coefficients)

    else:
        raise ValueError("no nles??")



if __name__ == '__main__':
    main()
