#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
try:
    from chainer_sklearn import SklearnWrapperClassifier, \
        SklearnWrapperRegressor
except ImportError:
    print('You need to install chainer_sklearn to run this example,\n'
          'please run `pip install -U chainer_sklearn`')

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print('You need to install scikit-learn to run this example,\n'
          'please run `pip install -U scikit-learn`')

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import chainer
from chainer import functions as F, cuda, Variable, serializers
import numpy

from chainer_chemistry.dataset.converters import concat_mols

from model import model_constructor
from data import prepare_qm9_dataset


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet']
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
    parser.add_argument('--conv_layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--unit_num', '-u', type=int, default=16)
    args = parser.parse_args()

    # data preparation
    seed = 777
    train_data_ratio = 0.7
    method = args.method
    train, val, class_num, ss = prepare_qm9_dataset(
        method, labels=args.label, train_data_ratio=train_data_ratio,
        seed=seed, scale=args.scale
    )

    # Construct Network
    model = model_constructor(method, class_num, args.unit_num,
                              args.conv_layers)

    def scaled_abs_error(x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        if args.scale == 'standardize':
            scaled_x0 = ss.inverse_transform(cuda.to_cpu(x0))
            scaled_x1 = ss.inverse_transform(cuda.to_cpu(x1))
            diff = scaled_x0 - scaled_x1
        elif args.scale == 'none':
            diff = cuda.to_cpu(x0) - cuda.to_cpu(x1)
        return numpy.mean(numpy.absolute(diff), axis=0)[0]

    classifier = SklearnWrapperRegressor(model,
                                         lossfun=F.mean_squared_error,
                                         accfun=scaled_abs_error,
                                         device=args.gpu)
    classifier.fit(train,
                   test=val,
                   batchsize=args.batchsize,
                   converter=concat_mols,
                   iterator_class=chainer.iterators.SerialIterator,
                   optimizer=chainer.optimizers.Adam(),
                   device=args.gpu,
                   epoch=args.epoch,
                   out=args.out,
                   snapshot_frequency=10,
                   dump_graph=False,
                   log_report=True,
                   plot_report=True,
                   print_report=True,
                   progress_report=True,
                   resume=args.resume)

    save_path = '{}/{}_{}.npz'.format(args.out, method, args.label)
    print('saving model to {}...'.format(save_path))
    serializers.save_npz(save_path, classifier)

if __name__ == '__main__':
    main()
