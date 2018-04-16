#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import pickle

from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
# from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass


# import chainer
# from chainer import functions as F, cuda, Variable
from chainer import cuda, Variable
# from chainer import iterators as I
# from chainer import optimizers as O
# from chainer import training
from chainer.datasets import split_dataset_random
# from chainer.training import extensions as E
import numpy

from chainer_chemistry import datasets as D
# from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN
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

# These import is necessary for pickle to work
from train_qm9 import GraphConvPredictor
from train_qm9 import ScaledAbsError


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
    # parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    # parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--in-dir', '-i', type=str, default='result')
    # parser.add_argument('--epoch', '-e', type=int, default=20)
    # parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--train-data-ratio', '-t', type=float, default=0.7)
    # parser.add_argument('--protocol', type=int, default=2)
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

    if os.path.exists(cache_dir):
        print('load from cache {}'.format(cache_dir))
        dataset = NumpyTupleDataset.load(os.path.join(cache_dir, 'data.npz'))
    if dataset is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        dataset = D.get_qm9(preprocessor, labels=labels)
        os.makedirs(cache_dir)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'data.npz'), dataset)

    if args.scale == 'standardize':
        # Standard Scaler for labels
        with open(os.path.join(args.in_dir, 'ss.pkl'), mode='rb') as f:
            ss = pickle.load(f)
    else:
        ss = None
        # ss = StandardScaler()
        # labels = ss.fit_transform(dataset.get_datasets()[-1])
        # dataset = NumpyTupleDataset(*dataset.get_datasets()[:-1], labels)

    train_data_size = int(len(dataset) * train_data_ratio)
    train, val = split_dataset_random(dataset, train_data_size, seed)

    # Network
    # n_unit = args.unit_num
    # conv_layers = args.conv_layers
    # if method == 'nfp':
    #     print('Train NFP model...')
    #     model = GraphConvPredictor(NFP(out_dim=n_unit, hidden_dim=n_unit,
    #                                    n_layers=conv_layers),
    #                                MLP(out_dim=class_num, hidden_dim=n_unit))
    # elif method == 'ggnn':
    #     print('Train GGNN model...')
    #     model = GraphConvPredictor(GGNN(out_dim=n_unit, hidden_dim=n_unit,
    #                                     n_layers=conv_layers),
    #                                MLP(out_dim=class_num, hidden_dim=n_unit))
    # elif method == 'schnet':
    #     print('Train SchNet model...')
    #     model = GraphConvPredictor(
    #         SchNet(out_dim=class_num, hidden_dim=n_unit, n_layers=conv_layers),
    #         None)
    # elif method == 'weavenet':
    #     print('Train WeaveNet model...')
    #     n_atom = 20
    #     n_sub_layer = 1
    #     weave_channels = [50] * conv_layers
    #     model = GraphConvPredictor(
    #         WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
    #                  n_sub_layer=n_sub_layer, n_atom=n_atom),
    #         MLP(out_dim=class_num, hidden_dim=n_unit))
    # elif method == 'rsgcn':
    #     print('Train RSGCN model...')
    #     model = GraphConvPredictor(
    #         RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
    #         MLP(out_dim=class_num, hidden_dim=n_unit))
    # else:
    #     raise ValueError('[ERROR] Invalid method {}'.format(method))
    regressor = Regressor.load_pickle(
        os.path.join(args.in_dir, 'regressor.pkl'),
        device=args.gpu)  # type: Regressor

    # train_iter = I.SerialIterator(train, args.batchsize)
    # val_iter = I.SerialIterator(val, args.batchsize,
    #                             repeat=False, shuffle=False)

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

    # We need to feed only input features `x` to `predict`/`predict_proba`.
    # This converter extracts only inputs (x1, x2, ...) from the features which
    # consist of input `x` and label `t` (x1, x2, ..., t).
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]

    def postprocess_fn(x):
        if ss is not None:
            # Model's output is scaled by StandardScaler,
            # so we need to rescale back.
            if isinstance(x, Variable):
                x = x.data
                scaled_x = ss.inverse_transform(cuda.to_cpu(x))
                return scaled_x
        else:
            return x

    y_pred = regressor.predict(val, converter=extract_inputs,
                               postprocess_fn=postprocess_fn)

    print('y_pred.shape = {}, y_pred[:5, 0] = {}'
          .format(y_pred.shape, y_pred[:5, 0]))

    # t = val.features[:, -1]
    t = concat_mols(val, device=-1)[-1]
    import IPython; IPython.embed()


    target_label = 0
    n_eval = 10
    for i in range(n_eval):
        print('i = {}, y_pred = {}, t = {}, diff = {}'
              .format(i, y_pred[i, target_label], t[i, target_label],
                      y_pred[i, target_label] - t[i, target_label]))

    # --- evaluate ---
    # To calc loss/accuracy, we can use `Evaluator`, `ROCAUCEvaluator`
    print('Evaluating...')
    val_iterator = SerialIterator(val, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(
        val_iterator, regressor, converter=concat_mols, device=args.gpu)()
    print('Evaluation result: ', eval_result)

    # regressor = Regressor(
    #     model, lossfun=F.mean_squared_error,
    #     metrics_fun={'abs_error': scaled_abs_error}, device=args.gpu)

    # optimizer = O.Adam()
    # optimizer.setup(regressor)

    # updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
    #                                    converter=concat_mols)
    # trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    # trainer.extend(E.Evaluator(val_iter, regressor, device=args.gpu,
    #                            converter=concat_mols))
    # trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    # trainer.extend(E.LogReport())
    # trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/abs_error',
    #                               'validation/main/loss',
    #                               'validation/main/abs_error',
    #                               'elapsed_time']))
    # trainer.extend(E.ProgressBar())
    # trainer.run()
    import IPython; IPython.embed()


if __name__ == '__main__':
    main()
