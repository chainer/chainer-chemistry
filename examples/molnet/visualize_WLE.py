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
        print(model)
        print(model.predictor)
        cnle = model.predictor.graph_conv
        print(cnle)
        concatW = cnle.linear_for_concat_nle.W.data
        print(type(concatW))

        [D, D2] = np.shape(concatW)

        #
        # visualize the raw W
        #
        out_prefix = out_prefix + "_learnedW"
        with open(out_prefix + ".dat", 'w') as fout:
            import csv
            writer = csv.writer(fout, lineterminator="\n")
            writer.writerows(concatW)
        # end with

        # dump the stat
        W1 = concatW[:, :D].flatten()
        W2 = concatW[:, D:].flatten()
        avgs = (np.mean(W1), np.mean(W2))
        stds = (np.std(W1), np.std(W2))        
        
        with open(out_prefix + ".stat", 'w') as fout:
            import csv
            fout.write("origin: " + str(avgs[0]) + " pm " + str(stds[0]) + "\n")
            fout.write("origin: " + str(avgs[1]) + " pm " + str(stds[1]) + "\n")
        # end with
        
        # visualize
        fig1, ax1 = plt.subplots()
        plt.imshow(concatW, cmap="jet")
        plt.colorbar(ax=ax1)

        plt.title('Learned W on ' + dataset_name + ' + ' + method)
        plt.savefig(out_prefix + ".png")
        plt.savefig(out_prefix + ".pdf")

        
        #
        # visualize the absolute value
        #
        fig2, ax2 = plt.subplots()
        plt.imshow(np.abs(concatW), cmap="binary")
        plt.colorbar(ax=ax2)

        plt.title('Learned abs(W) on ' + dataset_name + ' + ' + method)
        plt.savefig(out_prefix + "_abs.png")
        plt.savefig(out_prefix + "_abs.pdf")

        # dump the stat
        W1 = np.abs(W1)
        W2 = np.abs(W2)
        W = np.abs(concatW).flatten()
        W_mean = np.mean(W)
        W_std = np.std(W)
        # number of non-sparse element
        thre_high = W_mean + W_std
        thre_low = W_mean - W_std

        num_highs = ( len( np.where(W1 > thre_high)[0] ), len( np.where(W2 > thre_high)[0] ) )
        num_lows = ( len( np.where(W1 > thre_low)[0] ), len( np.where(W2 > thre_low)[0] ) )
        
        avgs = (np.mean(W1), np.mean(W2))
        stds = (np.std(W1), np.std(W2))

        # data and norms
        h_list = []
        h_s_list = []
        #h_norms = np.zeros( len(data) )
        #h_s_norms = np.zeros( len(data) )
        
        for i, data_i in enumerate(data):
            #print( len(data_i))
            atom_array = data_i[0]
            adj = data_i[1]
            nle_array = data_i[2]
            #print(atom_array)
            #print(adj)
            #print(nle_array)
            
            h = cnle.embed(atom_array).data
            h_list.append(h)
            h_s = cnle.embed_nle(nle_array).data
            h_s_list.append(h_s)
            #h_norms[i] = np.mean(np.linalg.norm(h, axis=1))
            #print("h_norm.shape=", h_norm.shape)
            #h_s_norms[i] = np.mean(np.linalg.norm(h_s, axis=1))
            #print("h_s_nomr.shape=", h_s_norm.shape)
        h = np.vstack(h_list)
        print("h.shape=", h.shape)
        h_s = np.vstack(h_s_list)
        print("h_s.shape=", h_s.shape)
        h_norms = np.linalg.norm(h, axis=1)
        print("h_norms.shape=", h_norms.shape)
        h_s_norms = np.linalg.norm(h_s, axis=1)
        print("h_s_norms.shape=", h_s_norms.shape)
                
        with open(out_prefix + "_abs.stat", 'w') as fout:
            import csv
            fout.write("origin: " + str(avgs[0]) + " pm " + str(stds[0]) + "\n")
            fout.write("WL: " + str(avgs[1]) + " pm " + str(stds[1]) + "\n")
            fout.write("all mean/std: " + str(W_mean) + "/" + str(W_std) + "\n")
            fout.write("origin num high, low: " + str(num_highs[0]) + ", " + str(num_lows[0]) + "\n")
            fout.write("WL num high, low: " + str(num_highs[1]) + ", " + str(num_lows[1]) + "\n")
            fout.write("origin norm: " + str(np.mean(h_norms)) + ", WL norm: " + str(np.mean(h_s_norms)) + "\n")
        # end with

        #
        # visualize the scaled weights (L2 embed=1)
        #
        fig3, ax3 = plt.subplots()
        concatW[:, :D] = concatW[:, :D] * np.mean(h_norms)
        concatW[:, D:] = concatW[:, D:] * np.mean(h_s_norms)        
        plt.imshow(np.abs(concatW), cmap="binary")
        plt.colorbar(ax=ax3)

        plt.title('Learned abs(W\_scaled) on ' + dataset_name + ' + ' + method)
        plt.savefig(out_prefix + "_abs_scaled.png")
        plt.savefig(out_prefix + "_abs_scaled.pdf")

        # dump the stat
        W1_scaled = W1 * np.mean(h_norms) # abs. values
        W2_scaled = W2 * np.mean(h_s_norms) # abs. values
        
        avgs = (np.mean(W1_scaled), np.mean(W2_scaled))
        stds = (np.std(W1_scaled), np.std(W2_scaled))

        # do statistical test
        
        with open(out_prefix + "_abs_scaled.stat", 'w') as fout:
            import csv
            fout.write("origin: " + str(avgs[0]) + " pm " + str(stds[0]) + "\n")
            fout.write("WL: " + str(avgs[1]) + " pm " + str(stds[1]) + "\n")
            fout.write("statistical test: " + str(avgs[1]) + " pm " + str(stds[1]) + "\n")
            
        # end with

        
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


    # Re-load the best-validation score snapshot
    # serializers.load_npz(os.path.join(
    #     model_dir, "best_val_" + model_filename[task_type]), model)

    # Run an evaluator on the test dataset.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
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
    save_json(os.path.join(model_dir, 'eval_result.json'), eval_result)


if __name__ == '__main__':
    main()
