#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import os

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

# These import is necessary for pickle to work
# from sklearn.preprocessing import StandardScaler  # NOQA
from train_molnet import GraphConvPredictor, GraphConvPredictorForGWM  # NOQA
from train_molnet import dataset_part_filename
from train_molnet import download_entire_dataset


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn', 'gin', 'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm']
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
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory to load model data from')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data

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
                                             method, cache_dir)

#    # Load the standard scaler parameters, if necessary.
#    if args.scale == 'standardize':
#        scaler_path = os.path.join(args.in_dir, 'scaler.pkl')
#        print('Loading scaler parameters from {}.'.format(scaler_path))
#        with open(scaler_path, mode='rb') as f:
#            scaler = pickle.load(f)
#    else:
#        print('No standard scaling was selected.')
#        scaler = None

    # Model-related data is stored this directory.
    model_dir = os.path.join(args.in_dir, os.path.basename(cache_dir))

    model_filename = {'classification': 'classifier.pkl',
                      'regression': 'regressor.pkl'}
    task_type = molnet_default_config[dataset_name]['task_type']
    model_path = os.path.join(model_dir, model_filename[task_type])
    print("model_path=" + model_path)
    print('Loading model weights from {}...'.format(model_path))

    if task_type == 'classification':
        model = Classifier.load_pickle(model_path, device=args.gpu)
    elif task_type == 'regression':
        model = Regressor.load_pickle(model_path, device=args.gpu)
    else:
        raise ValueError('Invalid task type ({}) encountered when processing '
                         'dataset ({}).'.format(task_type, dataset_name))

    # Proposed by Ishiguro
    # ToDo: consider go/no-go with following modification
    # Re-load the best-validation score snapshot
    serializers.load_npz(os.path.join(model_dir, "best_val_" + model_filename[task_type]), model)


#    # Replace the default predictor with one that scales the output labels.
#    scaled_predictor = ScaledGraphConvPredictor(model.predictor)
#    scaled_predictor.scaler = scaler
#    model.predictor = scaled_predictor

    # Run an evaluator on the test dataset.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, model, converter=concat_mols,
                            device=args.gpu)()
    print('Evaluation result: ', eval_result)


    # Proposed by Ishiguro: add more stats
    # ToDo: considre go/no-go with the following modification

    if task_type=='regression':
        #loss = cuda.to_cpu(numpy.array(eval_result['main/loss']))
        #eval_result['main/loss'] = loss

        # convert to native values..
        for k, v in eval_result.items():
            eval_result[k] = float(v)

        with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
            json.dump(eval_result, f)
        # end-with

    elif task_type=="classification":
        # For Classifier, we do not equip the model with ROC-AUC evalation function
        # use a seperate ROC-AUC Evaluator here
        rocauc_result = ROCAUCEvaluator(test_iterator, model, converter=concat_mols, device=args.gpu,eval_func=model.predictor, name='test', ignore_labels=-1)()
        print('ROCAUC Evaluation result: ', rocauc_result)
        with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
            json.dump(rocauc_result, f)
    else:
        pass


    # Save the evaluation results.
    with open(os.path.join(model_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
