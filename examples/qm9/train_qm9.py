#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy
import os

from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.links.scaler.standard_scaler import StandardScaler
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions.auto_print_report import \
    AutoPrintReport


class MeanAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) mean absolute error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if self.scaler is not None:
            x0 = self.scaler.inverse_transform(x0)
            x1 = self.scaler.inverse_transform(x1)
        return F.mean_absolute_error(x0, x1)


class RootMeanSqrError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) root mean square error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if self.scaler is not None:
            x0 = self.scaler.inverse_transform(x0)
            x1 = self.scaler.inverse_transform(x1)
        return F.sqrt(F.mean_squared_error(x0, x1))


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat']
    label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                   'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Regression on QM9.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp', help='method name')
    parser.add_argument('--label', '-l', type=str,
                        choices=label_names + ['all'], default='all',
                        help='target label for regression; all means '
                        'predicting all properties at once')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        default='standardize', help='label scaling method')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
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
    if args.label != 'all':
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_all'.format(method))
        class_num = len(D.get_qm9_label_names())

    # Get the filename corresponding to the cached dataset, based on the amount
    # of data samples that need to be parsed from the original dataset.
    num_data = args.num_data
    if num_data >= 0:
        dataset_filename = 'data_{}.npz'.format(num_data)
    else:
        dataset_filename = 'data.npz'

    # Load the cached dataset.
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)

    dataset = None
    if os.path.exists(dataset_cache_path):
        print('Loading cached dataset from {}.'.format(dataset_cache_path))
        dataset = NumpyTupleDataset.load(dataset_cache_path)
    if dataset is None:
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()

        if num_data >= 0:
            # Select the first `num_data` samples from the dataset.
            target_index = numpy.arange(num_data)
            dataset = D.get_qm9(preprocessor, labels=labels,
                                target_index=target_index)
        else:
            # Load the entire dataset.
            dataset = D.get_qm9(preprocessor, labels=labels)

        # Cache the laded dataset.
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)

    # Scale the label values, if necessary.
    if args.scale == 'standardize':
        print('Applying standard scaling to the labels.')
        scaler = StandardScaler()
        scaled_t = scaler.fit_transform(dataset.get_datasets()[-1])
        dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1]
                                      + (scaled_t,)))
    else:
        print('No standard scaling was selected.')
        scaler = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, valid = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(method, args.unit_num, args.conv_layers,
                                 class_num, scaler)

    # Set up the iterators.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False,
                                          shuffle=False)

    # Set up the regressor.
    device = args.gpu
    metrics_fun = {'mae': MeanAbsError(scaler=scaler),
                   'rmse': RootMeanSqrError(scaler=scaler)}
    regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                          metrics_fun=metrics_fun, device=device)

    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(regressor)

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=device,
                                       converter=concat_mols)

    # Set up the trainer.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(valid_iter, regressor, device=device,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(AutoPrintReport())
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))
    regressor.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    main()
