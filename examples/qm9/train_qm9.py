#!/usr/bin/env python
from __future__ import print_function

import argparse
import chainer
import numpy
import os
import pickle

from sklearn.preprocessing import StandardScaler

from chainer import cuda
from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E
from chainer import Variable

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import (
    MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN, RelGCN, RelGAT)
from chainer_chemistry.models.prediction import Regressor


class GraphConvPredictor(chainer.Chain):
    def __init__(self, graph_conv, mlp=None):
        """Initializes the graph convolution predictor.
        Args:
            graph_conv: The graph convolution network required to obtain
                        molecule feature representation.
            mlp: Multi layer perceptron; used as the final fully connected
                 layer. Set it to `None` if no operation is necessary
                 after the `graph_conv` calculation.
        """
        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs, is_real_node=None):
        if is_real_node is None:
            x = self.graph_conv(atoms, adjs)
        else:
            x = self.graph_conv(atoms, adjs, is_real_node)
        if self.mlp:
            x = self.mlp(x)
        return x


class MeanAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) mean absolute error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]


class RootMeanSqrError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) root mean square error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scaler is not None:
            scaled_x0 = self.scaler.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.scaler.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        else:
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.sqrt(numpy.mean(numpy.power(diff, 2), axis=0)[0])


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
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='',
                        help='target label for regression; empty string means '
                        'predicting all properties at once')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
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


def set_up_predictor(method, n_unit, conv_layers, class_num):
    """Sets up the predictor, consisting of a graph convolution network and
    a multilayer perceptron.

    Args:
        method: Method name. Currently, the supported ones are `nfp`, `ggnn`,
                `schnet`, `weavenet` and `rsgcn`.
        n_unit: Number of hidden units.
        conv_layers: Number of convolutional layers for the graph convolution
                     network.
        class_num: Number of output classes.
    Returns:
        An instance of the selected predictor.
    """

    predictor = None
    mlp = MLP(out_dim=class_num, hidden_dim=n_unit)

    if method == 'nfp':
        print('Training an NFP predictor...')
        nfp = NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(nfp, mlp)
    elif method == 'ggnn':
        print('Training a GGNN predictor...')
        ggnn = GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(ggnn, mlp)
    elif method == 'schnet':
        print('Training an SchNet predictor...')
        schnet = SchNet(out_dim=class_num, hidden_dim=n_unit,
                        n_layers=conv_layers)
        predictor = GraphConvPredictor(schnet, None)
    elif method == 'weavenet':
        print('Training a WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers

        weavenet = WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                            n_sub_layer=n_sub_layer, n_atom=n_atom)
        predictor = GraphConvPredictor(weavenet, mlp)
    elif method == 'rsgcn':
        print('Training an RSGCN predictor...')
        rsgcn = RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers)
        predictor = GraphConvPredictor(rsgcn, mlp)
    elif method == 'relgcn':
        print('Use Relational GCN predictor...')
        num_edge_type = 4
        relgcn = RelGCN(out_channels=class_num, num_edge_type=num_edge_type,
                        scale_adj=True)
        predictor = GraphConvPredictor(relgcn, None)
    elif method == 'relgat':
        print('Train Relational GAT predictor...')
        relgat = RelGAT(out_dim=n_unit, hidden_dim=n_unit,
                        n_layers=conv_layers)
        predictor = GraphConvPredictor(relgat, mlp)
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))
    return predictor


def main():
    # Parse the arguments.
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    method = args.method
    if args.label:
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
        os.makedirs(cache_dir, exist_ok=True)
        NumpyTupleDataset.save(dataset_cache_path, dataset)

    # Scale the label values, if necessary.
    if args.scale == 'standardize':
        print('Applying standard scaling to the labels.')
        scaler = StandardScaler()
        labels = scaler.fit_transform(dataset.get_datasets()[-1])
        dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1] + (labels,)))
        labels = scaler.fit_transform(dataset.get_datasets()[-1])
    else:
        print('No standard scaling was selected.')
        scaler = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, valid = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(method, args.unit_num, args.conv_layers,
                                 class_num)

    # Set up the iterators.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False,
                                          shuffle=False)

    # Set up the regressor.
    metrics_fun = {'mean_abs_error': MeanAbsError(scaler=scaler),
                   'root_mean_sqr_error': RootMeanSqrError(scaler=scaler)}
    regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                          metrics_fun=metrics_fun, device=args.gpu)

    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(regressor)

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)

    # Set up the trainer.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(valid_iter, regressor, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
                                  'validation/main/loss',
                                  'validation/main/abs_error',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, args.model_filename)
    print('Saving the trained model to {}...'.format(model_path))
    regressor.save_pickle(model_path, protocol=args.protocol)

    # Save the standard scaler's parameters.
    if scaler is not None:
        scaler_path = os.path.join(args.out, 'scaler.pkl')
        print('Saving standard scaler parameters to {}.'.format(scaler_path))
        with open(scaler_path, mode='wb') as f:
            pickle.dump(scaler, f, protocol=args.protocol)


if __name__ == '__main__':
    main()
