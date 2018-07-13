#!/usr/bin/env python

from __future__ import print_function

import chainer
import numpy
import os
import pickle
import sys

from argparse import ArgumentParser
from chainer.datasets import split_dataset_random
from chainer import functions as F, cuda, Variable
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions as E
from sklearn.preprocessing import StandardScaler

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN, Regressor  # NOQA


class GraphConvPredictor(chainer.Chain):
    def __init__(self, graph_conv, mlp=None):
        """Initialize GraphConvPredictor

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
    def __init__(self, scaler=None):
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


def set_up_predictor(method, n_unit, conv_layers, class_num):
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
    else:
        raise ValueError('[ERROR] Invalid method: {}'.format(method))

    return predictor


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on a custom dataset')
    parser.add_argument('--datafile', '-d', type=str, default='dataset.csv',
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
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
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
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        sys.exit("Error: No target label is specified.")

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
        ss = StandardScaler()
        labels = ss.fit_transform(dataset.get_datasets()[-1])
        dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1] + (labels,)))
    else:
        ss = None

    # Split the dataset into training and validation.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    train, val = split_dataset_random(dataset, train_data_size, args.seed)

    # Set up the predictor.
    predictor = set_up_predictor(args.method, args.unit_num,
                                 args.conv_layers, class_num)

    # Set up the iterators.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(val, args.batchsize, repeat=False,
                                        shuffle=False)

    # Set up the regressor.
    regressor = Regressor(predictor, lossfun=F.mean_squared_error,
                          metrics_fun={'abs_error': ScaledAbsError(scaler=ss)},
                          device=args.gpu)

    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(regressor)

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)

    # Set up the trainer.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(val_iter, regressor, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
                                  'validation/main/loss',
                                  'validation/main/abs_error', 'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the regressor's parameters.
    model_path = os.path.join(args.out, 'model.npz')
    print('saving trained model to {}'.format(model_path))
    serializers.save_npz(model_path, regressor)

    # Save the standard scaler's parameters.
    if ss is not None:
        with open(os.path.join(args.out, 'scaler.pkl'), mode='wb') as f:
            pickle.dump(ss, f, protocol=args.protocol)


if __name__ == '__main__':
    main()
