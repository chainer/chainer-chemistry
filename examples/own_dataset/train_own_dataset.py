#!/usr/bin/env python

from __future__ import print_function

import chainer
import numpy
import os

from argparse import ArgumentParser
from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer import optimizers
from chainer import training
from chainer.iterators import SerialIterator
from chainer.training import extensions as E

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.models import Regressor
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions.auto_print_report import \
    AutoPrintReport
from chainer_chemistry.utils import run_train


def rmse(x0, x1):
    return F.sqrt(F.mean_squared_error(x0, x1))


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'mpnn', 'gnnfilm']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
    parser.add_argument('--datafile', '-d', type=str,
                        default='dataset_train.csv',
                        help='csv file containing the dataset')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['value1', 'value2'],
                        help='target label for regression')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument(
        '--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=10,
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
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation. Postprocessing is required for the regression task.
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)

    # Apply a preprocessor to the dataset.
    print('Preprocessing dataset...')
    preprocessor = preprocess_method_dict[args.method]()
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    dataset = parser.parse(args.datafile)['dataset']

    # Scale the label values, if necessary.
    if args.scale == 'standardize':
        scaler = StandardScaler()
        scaler.fit(dataset.get_datasets()[-1])
    else:
        scaler = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, _ = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(
        args.method, args.unit_num,
        args.conv_layers, class_num, label_scaler=scaler)

    # Set up the regressor.
    device = chainer.get_device(args.device)
    metrics_fun = {'mae': F.mean_absolute_error, 'rmse': rmse}
    regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                          metrics_fun=metrics_fun, device=device)

    print('Training...')
    run_train(regressor, train, valid=None,
              batch_size=args.batchsize, epoch=args.epoch,
              out=args.out, extensions_list=None,
              device=device, converter=concat_mols,
              resume_path=None)

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))

    # TODO(nakago): ChainerX array cannot be sent to numpy array when internal
    # state has gradients.
    if hasattr(regressor.predictor.graph_conv, 'reset_state'):
        regressor.predictor.graph_conv.reset_state()

    regressor.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    main()
