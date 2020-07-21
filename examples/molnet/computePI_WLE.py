#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os

import numpy as np

import chainer
from chainer import functions
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
                   'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm',
                   'nfp_cnle', 'ggnn_cnle', 'relgat_cnle', 'relgcn_cnle', 'rsgcn_cnle', 'gin_cnle',
                   'nfp_gcnle', 'ggnn_gcnle', 'relgat_gcnle', 'relgcn_gcnle', 'rsgcn_gcnle', 'gin_gcnle']
#    scale_list = ['standardize', 'none']
    dataset_names = list(molnet_default_config.keys())

    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Prediction on Molnet.')
    parser.add_argument('--dataset', '-d', type=str, choices=dataset_names,
                        default='bbbp',
                        help='name of the dataset that training is run on')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
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


def shuffle_testset(test):
    """

    :param test:
    :return: two datasets
        (i) test set where atom array is shuffled
        (ii) test set where nle array is shuffled
    """

    print("type(test)=", type(test)) # chainer_chemistry.datasets.numpy_tuple_dataset.NumpyTupleDataset

    # break the TupleDataset

    atom_arrays = []  # Mol by N
    adj_arrays = []
    nle_arrays = []  # Mol by N by N or N by R by N by N
    ys = []  # Mol by (data-dependent)

    shuffled_atom_arrays = []  # Mol by N
    shuffled_nle_arrays = []  # Mol by N by N or N by R by N by N

    for mol_data in test:

        atom_array = mol_data[0]
        adj_array = mol_data[1]
        nle_array = mol_data[2]
        y = mol_data[3]

        if False:
            print("type(mol_data)=", type(mol_data), ", len(mol_dat)=", len(mol_data)) # tuple
            print("type(atom_arrray)=", type(atom_array), ", shape(atom_array)=", np.shape(atom_array)) #ndarray
            print("type(adj_arrray)=", type(adj_array), ", shape(adj_array)=", np.shape(adj_array)) # ndarray
            print("type(nle_arrray)=", type(nle_array), ", shape(nle_array)=", np.shape(nle_array)) # ndarray
            print("type(y)=", type(y), ", len(y)=", len(y)) # ndarray

        # choose an atom to shuffle
        i = np.floor(0.5 * np.random.rand(1) * len(atom_array)).astype(int)
        #print("type(i)=", type(i), ", i=", i)

        # copy shuffled buffer
        shuffled_atom_array = np.copy(atom_array)
        shuffled_nle_array = np.copy(nle_array)

        shuffled_atom_array[i] = atom_array[i] - 1 if atom_array[i] > 0 else atom_array[i] + 1
        shuffled_nle_array[i] = nle_array[i] - 1 if nle_array[i] > 0 else nle_array[i] + 1

        #print(atom_array)
        #print(shuffled_atom_array)

        atom_arrays.append(atom_array)
        adj_arrays.append(adj_array)
        nle_arrays.append(nle_array)
        ys.append(y)

        shuffled_atom_arrays.append(shuffled_atom_array)
        shuffled_nle_arrays.append(shuffled_nle_array)

    test_atom_shuffled = NumpyTupleDataset(shuffled_atom_arrays, adj_arrays, nle_arrays, ys)
    test_nle_shuffled = NumpyTupleDataset(atom_arrays, adj_arrays, shuffled_nle_arrays, ys)

    return test_atom_shuffled, test_nle_shuffled

def main():
    """
    Load the dataset and the learned WLE model.
    Then compute the permutation feature importance.

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

    # generarte shuffled test set
    test_atom_shuffled, test_nle_shuffled = shuffle_testset(test)

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
    test_atom_shuffled_iterator = SerialIterator(test_atom_shuffled, args.batchsize, repeat=False, shuffle=False)
    test_nle_shuffled_iterator = SerialIterator(test_nle_shuffled, args.batchsize, repeat=False, shuffle=False)


    # Re-load the best-validation score snapshot
    # serializers.load_npz(os.path.join(
    #     model_dir, "best_val_" + model_filename[task_type]), model)

    # Run an evaluator on the test dataset.
    print('Evaluating the correct test set')
    eval_result = Evaluator(test_iterator, model, converter=concat_mols,
                            device=device)()
    print('Evaluation result: ', eval_result)

    # Add more stats
    if task_type == 'regression':
        # loss = cuda.to_cpu(numpy.array(eval_result['main/loss']))
        # eval_result['main/loss'] = loss

        # convert to native values..
        for k, v in eval_result.items():
            eval_result[k] = float(v)

    elif task_type == "classification":
        # For Classifier, we do not equip the model with ROC-AUC evalation
        # function. use separate ROC-AUC Evaluator
        rocauc_result = ROCAUCEvaluator(
            test_iterator, model, converter=concat_mols, device=device,
            eval_func=model.predictor, name='test', ignore_labels=-1)()
        print('ROCAUC Evaluation result: ', rocauc_result)

        # add
        for k, v in rocauc_result.items():
            eval_result[k] = float(v)
        #save_json(os.path.join(model_dir, 'rocauc_result.json'), rocauc_result)
    else:
        print('[WARNING] unknown task_type {}.'.format(task_type))

    # Save the evaluation results.
    # Do not overwrite the original result save_json(os.path.join(model_dir, 'eval_result.json'), eval_result)

    # Run an evaluator on the test dataset, atom shuffled.
    print('Evaluating the atom-shuffled test set')
    eval_result = Evaluator(test_atom_shuffled_iterator, model, converter=concat_mols,
                            device=device)()
    print('Evaluation result: ', eval_result)

    # Add more stats
    if task_type == 'regression':
        # loss = cuda.to_cpu(numpy.array(eval_result['main/loss']))
        # eval_result['main/loss'] = loss

        # convert to native values..
        for k, v in eval_result.items():
            eval_result[k] = float(v)

    elif task_type == "classification":
        # For Classifier, we do not equip the model with ROC-AUC evalation
        # function. use separate ROC-AUC Evaluator
        rocauc_result = ROCAUCEvaluator(
            test_atom_shuffled_iterator, model, converter=concat_mols, device=device,
            eval_func=model.predictor, name='test', ignore_labels=-1)()
        print('ROCAUC Evaluation result: ', rocauc_result)

        # add
        for k, v in rocauc_result.items():
            eval_result[k] = float(v)
        #save_json(os.path.join(model_dir, 'rocauc_result.json'), rocauc_result)
    else:
        print('[WARNING] unknown task_type {}.'.format(task_type))

    # Save the evaluation results.
    save_json(os.path.join(model_dir, 'eval_result_atom_shuffled.json'), eval_result)

    # Run an evaluator on the test dataset, NLE shuffled.
    print('Evaluating the NLE-shuffled test set')
    eval_result = Evaluator(test_nle_shuffled_iterator, model, converter=concat_mols,
                            device=device)()
    print('Evaluation result: ', eval_result)

    # Add more stats
    if task_type == 'regression':
        # loss = cuda.to_cpu(numpy.array(eval_result['main/loss']))
        # eval_result['main/loss'] = loss

        # convert to native values..
        for k, v in eval_result.items():
            eval_result[k] = float(v)

    elif task_type == "classification":
        # For Classifier, we do not equip the model with ROC-AUC evalation
        # function. use separate ROC-AUC Evaluator
        rocauc_result = ROCAUCEvaluator(
            test_nle_shuffled_iterator, model, converter=concat_mols, device=device,
            eval_func=model.predictor, name='test', ignore_labels=-1)()
        print('ROCAUC Evaluation result: ', rocauc_result)

        # add
        for k, v in rocauc_result.items():
            eval_result[k] = float(v)
        #save_json(os.path.join(model_dir, 'rocauc_result.json'), rocauc_result)
    else:
        print('[WARNING] unknown task_type {}.'.format(task_type))

    # Save the evaluation results.
    save_json(os.path.join(model_dir, 'eval_result_nle_shuffled.json'), eval_result)


if __name__ == '__main__':
    main()
