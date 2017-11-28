#!/usr/bin/env python

from __future__ import print_function

import os

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
from chainer import functions as F, cuda
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E
import numpy
from rdkit import RDLogger

from chainerchem import datasets as D
from chainerchem.models import MLP, NFP, GGNN, SchNet
from chainerchem.dataset.preprocessors import preprocess_method_dict
from chainerchem.dataset.converters import concat_mols
from chainerchem.datasets.numpy_tuple_dataset import NumpyTupleDataset


# Disable errors by RDKit occurred in preprocessing Tox21 dataset.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv, mlp):
        """Constructor

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
            atoms, adjs = concat_mols(data[i:i + batchsize], device=device)[:2]
            y = self._predict(atoms, adjs)
            y_list.append(cuda.to_cpu(y.data))
        y_array = numpy.concatenate(y_list, axis=0)
        return y_array


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet']
    label_names = D.get_tox21_label_names()

    parser = argparse.ArgumentParser(
        description='Multitask Learning with Tox21.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp', help='graph convolution model to use')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='', help='target label for logistic regression. '
                        'Use all labels if this option is not specified.')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                        'not to use GPU and to run the code in CPU.')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to output directory')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')
    args = parser.parse_args()

    method = args.method
    if args.label:
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_all'.format(method))
        class_num = len(label_names)

    # Dataset preparation
    train, val, test = None, None, None
    if os.path.exists(cache_dir):
        print('load from cache {}'.format(cache_dir))
        train = NumpyTupleDataset.load(os.path.join(cache_dir, 'train.npz'))
        val = NumpyTupleDataset.load(os.path.join(cache_dir, 'val.npz'))
        test = NumpyTupleDataset.load(os.path.join(cache_dir, 'test.npz'))
    if train is None or val is None or test is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        train, val, test = D.get_tox21(preprocessor, labels=labels)
        os.makedirs(cache_dir)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'train.npz'), train)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'val.npz'), val)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'test.npz'), test)

    # Network
    n_unit = args.unit_num
    if method == 'nfp':
        print('Train NFP model...')
        model = GraphConvPredictor(NFP(n_unit, n_unit, args.conv_layers),
                                   MLP(n_unit, class_num))
    elif method == 'ggnn':
        print('Train GGNN model...')
        model = GraphConvPredictor(GGNN(n_unit, n_unit, args.conv_layers),
                                   MLP(args.unit_num, class_num))
    elif method == 'schnet':
        print('Train SchNet model...')
        model = SchNet(n_unit, class_num, args.conv_layers, n_unit)
    else:
        print('[ERROR] Invalid mode')
        exit()

    train_iter = I.SerialIterator(train, args.batchsize)
    val_iter = I.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    classifier = L.Classifier(model,
                              lossfun=F.sigmoid_cross_entropy,
                              accfun=F.binary_accuracy)
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
    trainer.extend(E.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == '__main__':
    main()
