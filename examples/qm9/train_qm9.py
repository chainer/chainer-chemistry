#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import pickle

from sklearn.preprocessing import StandardScaler

import chainer
from chainer import cuda
from chainer.datasets import split_dataset_random
from chainer import functions as F
from chainer import iterators as I
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E
from chainer import Variable
import numpy

from chainer_chemistry import datasets as D
from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN  # NOQA
try:
    from chainer_chemistry.models.prediction import Regressor
except ImportError:
    print('[ERROR] This example uses newly implemented `Regressor` class.\n'
          'Please install the library from master branch.\n See '
          'https://github.com/pfnet-research/chainer-chemistry#installation'
          ' for detail.')
    exit()
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset


class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp=None):
        """Graph Convolution Predictor

        It sequentially combines graph convolution network and multi layer
        perceptron.
        `graph_conv` gathers the each node's feature to extract graph feature,
        `mlp` processed the extracted graph feature to calculate final output.

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer.
                It can be `None` if no operation is necessary after
                `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        if self.mlp:
            x = self.mlp(x)
        return x


class ScaledAbsError(object):

    def __init__(self, scale='standardize', ss=None):
        self.scale = scale
        self.ss = ss

    def __call__(self, x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if self.scale == 'standardize':
            scaled_x0 = self.ss.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = self.ss.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        elif self.scale == 'none':
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                   'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    scale_list = ['standardize', 'none']

    parser = argparse.ArgumentParser(
        description='Regression with QM9.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='', help='target label for regression, '
                                         'empty string means to predict all '
                                         'property at once')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        default='standardize', help='Label scaling method')
    parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--train-data-ratio', '-t', type=float, default=0.7)
    parser.add_argument('--protocol', type=int, default=2)
    parser.add_argument('--model-filename', type=str, default='regressor.pkl')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='Number of data to be parsed from parser.'
                             '-1 indicates to parse all data.')
    args = parser.parse_args()

    seed = args.seed
    train_data_ratio = args.train_data_ratio
    method = args.method
    if args.label:
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_all'.format(method))
        class_num = len(D.get_qm9_label_names())

    # Dataset preparation
    dataset = None

    num_data = args.num_data
    if num_data >= 0:
        dataset_filename = 'data_{}.npz'.format(num_data)
    else:
        dataset_filename = 'data.npz'
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)
    if os.path.exists(dataset_cache_path):
        print('load from cache {}'.format(dataset_cache_path))
        dataset = NumpyTupleDataset.load(dataset_cache_path)
    if dataset is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        if num_data >= 0:
            # only use first 100 for debug
            target_index = numpy.arange(num_data)
            dataset = D.get_qm9(preprocessor, labels=labels,
                                target_index=target_index)
        else:
            dataset = D.get_qm9(preprocessor, labels=labels)
        os.makedirs(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)

    if args.scale == 'standardize':
        # Standard Scaler for labels
        ss = StandardScaler()
        labels = ss.fit_transform(dataset.get_datasets()[-1])
    else:
        ss = None
    dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1] + (labels,)))

    train_data_size = int(len(dataset) * train_data_ratio)
    train, val = split_dataset_random(dataset, train_data_size, seed)

    # Network
    n_unit = args.unit_num
    conv_layers = args.conv_layers
    if method == 'nfp':
        print('Train NFP model...')
        model = GraphConvPredictor(NFP(out_dim=n_unit, hidden_dim=n_unit,
                                       n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Train GGNN model...')
        model = GraphConvPredictor(GGNN(out_dim=n_unit, hidden_dim=n_unit,
                                        n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Train SchNet model...')
        model = GraphConvPredictor(
            SchNet(out_dim=class_num, hidden_dim=n_unit, n_layers=conv_layers),
            None)
    elif method == 'weavenet':
        print('Train WeaveNet model...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        model = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'rsgcn':
        print('Train RSGCN model...')
        model = GraphConvPredictor(
            RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid method {}'.format(method))

    train_iter = I.SerialIterator(train, args.batchsize)
    val_iter = I.SerialIterator(val, args.batchsize,
                                repeat=False, shuffle=False)

    regressor = Regressor(
        model, lossfun=F.mean_squared_error,
        metrics_fun={'abs_error': ScaledAbsError(scale=args.scale, ss=ss)},
        device=args.gpu)

    optimizer = O.Adam()
    optimizer.setup(regressor)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(val_iter, regressor, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
                                  'validation/main/loss',
                                  'validation/main/abs_error',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # --- save regressor & standardscaler ---
    protocol = args.protocol
    regressor.save_pickle(os.path.join(args.out, args.model_filename),
                          protocol=protocol)
    if args.scale == 'standardize':
        with open(os.path.join(args.out, 'ss.pkl'), mode='wb') as f:
            pickle.dump(ss, f, protocol=protocol)


if __name__ == '__main__':
    main()
