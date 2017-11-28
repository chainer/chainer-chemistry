#!/usr/bin/env python

from __future__ import print_function

import os

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print('You need to install scikit-learn to run this example,'
          'please run `pip install -U scikit-learn`')

from chainerchem.datasets import NumpyTupleDataset

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import numpy
import argparse
import chainer
from chainer import functions as F, cuda
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E

from chainerchem import datasets as D
from chainerchem.models import MLP, NFP, GGNN
from chainerchem.dataset.converters import concat_mols


class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp):
        """
        
        Args:
            graph_conv: graph convolution network to obtain molecule feature 
                        representation
            mlp: multi layer perceptron, used as final connected layer
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        x = self.mlp(x)
        return x

    def _predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)

    def predict(self, *args, batchsize=32, device=-1):
        if device >= 0:
            chainer.cuda.get_device_from_id(device).use()
            self.to_gpu()  # Copy the model to the GPU

        # TODO: Not test yet, check behavior
        data = args[0]
        y_list = []
        for i in range(0, len(data), batchsize):
            #adj, atom_types = concat_examples(data[i:i + batchsize], device=device)
            atoms, adjs = concat_mols(data[i:i + batchsize], device=device)[:2]
            y = self._predict(atoms, adjs)
            y_list.append(cuda.to_cpu(y.data))
        y_array = numpy.concatenate(y_list, axis=0)
        return y_array


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn']
    label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                   'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    parser = argparse.ArgumentParser(
        description='Regression with QM9.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='lumo', help='target label for regression')
    parser.add_argument('--scale', type=str, default='standardize',
                        help='Label scaling method')
    parser.add_argument('--conv_layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--unit_num', '-u', type=int, default=16)
    args = parser.parse_args()

    seed = 777
    train_data_ratio = 0.7
    method = args.method
    labels = args.label

    # Dataset preparation
    dataset = None
    cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
    #cache_dir = os.path.join('input', '{}'.format(method))
    if os.path.exists(cache_dir):
        print('load from cache {}'.format(cache_dir))
        dataset = NumpyTupleDataset.load(os.path.join(cache_dir, 'data.npz'))
    if dataset is None:
        print('preprocessing dataset...')
        dataset = D.get_qm9(preprocess_method=method, labels=labels)
        os.makedirs(cache_dir)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'data.npz'), dataset)

    train_data_size = int(len(dataset) * train_data_ratio)
    #train, val = split_dataset_random(dataset, train_data_size, seed)
    #from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
    #KFold(n_splits=1).get_n_splits()

    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(numpy.arange(len(dataset)),
                                          test_size=0.20, random_state=seed)
    print('train_idx', len(train_idx), train_idx)
    print('val_idx', len(val_idx), val_idx)

    #class_num = len(D.get_qm9_label_names())
    class_num = 1

    # Standard Scaler for labels
    if args.scale == 'standardize':
        # currently `features` indexer other than NumpyTupleDataset is on-going
        # PR, but not merged to chainer yet.
        train_labels = dataset.features[train_idx, -1]
        val_labels = dataset.features[val_idx, -1]
        ss = StandardScaler()
        train_labels = ss.fit_transform(train_labels)
        val_labels = ss.transform(val_labels)
        train = NumpyTupleDataset(*dataset.features[train_idx, :-1], train_labels)
        val = NumpyTupleDataset(*dataset.features[val_idx, :-1], val_labels)
    else:
        train = NumpyTupleDataset(*dataset.features[train_idx, :])
        val = NumpyTupleDataset(*dataset.features[val_idx, :])

    # Network
    if method == 'nfp':
        print('Train NFP model...')
        n_unit = args.unit_num
        model = GraphConvPredictor(NFP(n_unit, n_unit, args.conv_layers),
                                   MLP(n_unit, class_num))
    elif method == 'ggnn':
        print('Train GGNN model...')
        n_unit = args.unit_num
        model = GraphConvPredictor(GGNN(n_unit, n_unit, args.conv_layers),
                                   MLP(n_unit, class_num))
    else:
        print('[ERROR] Invalid mode')
        exit()

    train_iter = I.SerialIterator(train, args.batchsize)
    val_iter = I.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

    classifier = L.Classifier(model, lossfun=F.mean_squared_error,
                              accfun=F.absolute_error)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(classifier)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(val_iter, classifier, device=args.gpu, converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                  'validation/main/loss',
                                  'validation/main/accuracy',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()
