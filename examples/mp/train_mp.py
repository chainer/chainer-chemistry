#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy
import os


import chainer
from chainer import functions as F
from chainer.datasets import split_dataset_random


from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets.mp import MPDataset
from chainer_chemistry.links import StandardScaler
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.utils import run_train


def rmse(x0, x1):
    return F.sqrt(F.mean_squared_error(x0, x1))


def parse_arguments():
    label_names = ['formation_energy_per_atom', 'energy', 'band_gap', 'efermi',
                   'K_VRH', 'G_VRH', 'poisson_ratio']
    # Lists of supported preprocessing methods/models.
    method_list = ['megnet', 'cgcnn']
    scale_list = ['standardize', 'none']
    # Set up the argument parser.
    parser = argparse.ArgumentParser(
        description='Regression on Material Project Data.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='megnet', help='method name')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='formation_energy_per_atom',
                        help='target label for regression')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        default='standardize', help='label scaling method')
    parser.add_argument('--conv-layers', '-c', type=int, default=3,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument(
        '--device', '-d', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.7,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    method = args.method
    labels = args.label
    target_list = [labels]
    class_num = len(labels) if isinstance(labels, list) else 1
    cache_dir = os.path.join('input', '{}_{}'.format(method, labels))

    # Get the filename corresponding to the cached dataset, based on the amount
    # of data samples that need to be parsed from the original dataset.
    num_data = args.num_data
    if num_data >= 0:
        dataset_filename = 'data_{}.npz'.format(num_data)
    else:
        dataset_filename = 'data.npz'

    # Load the cached dataset.
    preprocessor = preprocess_method_dict[method]()
    dataset = MPDataset(preprocessor=preprocessor)
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)
    result = dataset.load_pickle(dataset_cache_path)

    # load datasets from Material Project Database
    if result is False:
        dataset.get_mp(target_list, args.num_data)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        dataset.save_pickle(dataset_cache_path)

    # Scale the label values, if necessary.
    if args.scale == 'standardize':
        print('Fit StandardScaler to the labels.')
        targets = numpy.concatenate([d[-1] for d in dataset.data], axis=0)
        scaler = StandardScaler()
        scaler.fit(targets)
    else:
        print('No standard scaling was selected.')
        scaler = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, valid = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(method, args.unit_num, args.conv_layers,
                                 class_num, scaler)

    # Set up the regressor.
    device = chainer.get_device(args.device)
    metrics_fun = {'mae': F.mean_absolute_error, 'rmse': rmse}
    regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                          metrics_fun=metrics_fun, device=device)

    print('Training...')
    converter = converter_method_dict[method]
    run_train(regressor, train, valid=valid,
              batch_size=args.batchsize, epoch=args.epoch,
              out=args.out, extensions_list=None,
              device=device, converter=converter,
              resume_path=None)

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))
    regressor.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    main()
