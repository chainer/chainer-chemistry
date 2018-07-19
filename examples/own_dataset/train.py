#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import pickle
import sys

import chainer
from chainer.datasets import split_dataset_random
from chainer import functions as F, cuda, Variable  # NOQA
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions as E
import numpy
from rdkit import Chem
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


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    scale_list = ['standardize', 'none']

    parser = argparse.ArgumentParser(
        description='Regression with own dataset.')
    parser.add_argument('--datafile', type=str, default='dataset.csv')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp')
    parser.add_argument('--label', '-l', nargs='+',
                        default=['value1', 'value2'],
                        help='target label for regression')
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
    args = parser.parse_args()

    seed = args.seed
    train_data_ratio = args.train_data_ratio
    method = args.method
    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        sys.exit("Error: No target label is specified.")

    # Dataset preparation
    # Postprocess is required for regression task
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)

    print('Preprocessing dataset...')
    preprocessor = preprocess_method_dict[method]()
    parser = CSVFileParser(preprocessor,
                           postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    dataset = parser.parse(args.datafile)["dataset"]

    if args.scale == 'standardize':
        # Standard Scaler for labels
        scaler = StandardScaler()
        labels = scaler.fit_transform(dataset.get_datasets()[-1])
        dataset = NumpyTupleDataset(*(dataset.get_datasets()[:-1] + (labels,)))
    else:
        # Not use scaler
        scaler = None

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

    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)

    regressor = Regressor(
        model, lossfun=F.mean_squared_error,
        metrics_fun={'abs_error': ScaledAbsError(scaler=scaler)},
        device=args.gpu)

    optimizer = optimizers.Adam()
    optimizer.setup(regressor)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(val_iter, regressor, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    # Note that original scale absolute errors are reported in
    # (validation/)main/abs_error
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
                                  'validation/main/loss',
                                  'validation/main/abs_error',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # --- save regressor's parameters ---
    protocol = args.protocol
    model_path = os.path.join(args.out, 'model.npz')
    print('saving trained model to {}'.format(model_path))
    serializers.save_npz(model_path, regressor)
    if scaler is not None:
        with open(os.path.join(args.out, 'scaler.pkl'), mode='wb') as f:
            pickle.dump(scaler, f, protocol=protocol)

    # Example of prediction using trained model
    smiles = 'c1ccccc1'
    mol = Chem.MolFromSmiles(smiles)
    preprocessor = preprocess_method_dict[method]()
    standardized_smiles, mol = preprocessor.prepare_smiles_and_mol(mol)
    input_features = preprocessor.get_input_features(mol)
    atoms, adjs = concat_mols([input_features], device=args.gpu)
    prediction = model(atoms, adjs).data[0]
    print('Prediction for {}:'.format(smiles))
    for i, label in enumerate(args.label):
        print('{}: {}'.format(label, prediction[i]))


if __name__ == '__main__':
    main()
