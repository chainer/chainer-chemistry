#!/usr/bin/env python

from __future__ import print_function

import os

import logging

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
from chainer import functions as F
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.training import extensions as E
import json
from rdkit import RDLogger

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry import datasets as D
try:
    from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator  # NOQA
except ImportError:
    print('[WARNING] If you want to use BalancedSerialIterator, please install'
          'the library from master branch.\n          See '
          'https://github.com/pfnet-research/chainer-chemistry#installation'
          ' for detail.')

import data
import predictor


# Disable errors by RDKit occurred in preprocessing Tox21 dataset.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
# show INFO level log from chainer chemistry
logging.basicConfig(level=logging.INFO)


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn',
                   'sparse_rsgcn']
    label_names = D.get_tox21_label_names()
    iterator_type = ['serial', 'balanced']

    parser = argparse.ArgumentParser(
        description='Multitask Learning with Tox21.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp', help='graph convolution model to use '
                        'as a predictor.')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='', help='target label for logistic '
                        'regression. Use all labels if this option '
                        'is not specified.')
    parser.add_argument('--iterator-type', type=str, choices=iterator_type,
                        default='serial', help='iterator type. If `balanced` '
                        'is specified, data is sampled to take same number of'
                        'positive/negative labels during training.')
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
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='path to a trainer snapshot')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    args = parser.parse_args()

    method = args.method
    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        class_num = len(label_names)

    # Dataset preparation
    train, val, _ = data.load_dataset(method, labels)

    # Network
    predictor_ = predictor.build_predictor(
        method, args.unit_num, args.conv_layers, class_num)

    iterator_type = args.iterator_type
    if iterator_type == 'serial':
        train_iter = I.SerialIterator(train, args.batchsize)
    elif iterator_type == 'balanced':
        if class_num > 1:
            raise ValueError('BalancedSerialIterator can be used with only one'
                             'label classification, please specify label to'
                             'be predicted by --label option.')
        train_iter = BalancedSerialIterator(
            train, args.batchsize, train.features[:, -1], ignore_labels=-1)
        train_iter.show_label_stats()
    else:
        raise ValueError('Invalid iterator type {}'.format(iterator_type))
    val_iter = I.SerialIterator(val, args.batchsize,
                                repeat=False, shuffle=False)
    classifier = L.Classifier(predictor_,
                              lossfun=F.sigmoid_cross_entropy,
                              accfun=F.binary_accuracy)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(classifier)

    padding = 0
    if method == 'sparse_rsgcn':
        padding = (0, 0, -1, -1, 0)  # atom, adj_data, adj_row, adj_col, label

    def converter(batch, device=None):
        return concat_mols(batch, device, padding)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(val_iter, classifier,
                               device=args.gpu, converter=converter))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                  'validation/main/loss',
                                  'validation/main/accuracy',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar(update_interval=10))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(E.snapshot(), trigger=(frequency, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    config = {'method': args.method,
              'conv_layers': args.conv_layers,
              'unit_num': args.unit_num,
              'labels': args.label}

    with open(os.path.join(args.out, 'config.json'), 'w') as o:
        o.write(json.dumps(config))


if __name__ == '__main__':
    main()
