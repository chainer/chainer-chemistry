#!/usr/bin/env python

from __future__ import print_function
import argparse
import json
import os
import pickle

from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
import pandas

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

from chainer import cuda
from chainer.datasets import split_dataset_random
from chainer import Variable
import numpy  # NOQA

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models.prediction import Regressor

# These import is necessary for pickle to work
from sklearn.preprocessing import StandardScaler  # NOQA
from train_qm9 import GraphConvPredictor  # NOQA
from train_qm9 import MeanAbsError, RootMeanSqrError  # NOQA


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
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--in-dir', '-i', type=str, default='result')
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--train-data-ratio', '-t', type=float, default=0.7)
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
        # class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = D.get_qm9_label_names()
        cache_dir = os.path.join('input', '{}_all'.format(method))
        # class_num = len(labels)

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
        dataset = D.get_qm9(preprocessor, labels=labels)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)

    if args.scale == 'standardize':
        # Standard Scaler for labels
        with open(os.path.join(args.in_dir, 'ss.pkl'), mode='rb') as f:
            ss = pickle.load(f)
    else:
        ss = None

    train_data_size = int(len(dataset) * train_data_ratio)
    train, test = split_dataset_random(dataset, train_data_size, seed)

    regressor = Regressor.load_pickle(
        os.path.join(args.in_dir, args.model_filename), device=args.gpu)

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

    print('Predicting...')
    y_pred = regressor.predict(test, converter=extract_inputs,
                               postprocess_fn=postprocess_fn)

    print('y_pred.shape = {}, y_pred[:5, 0] = {}'
          .format(y_pred.shape, y_pred[:5, 0]))

    t = concat_mols(test, device=-1)[-1]
    n_eval = 10

    # Construct dataframe
    df_dict = {}
    for i, l in enumerate(labels):
        df_dict.update({
            'y_pred_{}'.format(l): y_pred[:, i],
            't_{}'.format(l): t[:, i],
        })
    df = pandas.DataFrame(df_dict)

    # Show random 5 example's prediction/ground truth table
    print(df.sample(5))

    for target_label in range(y_pred.shape[1]):
        diff = y_pred[:n_eval, target_label] - t[:n_eval, target_label]
        print('target_label = {}, y_pred = {}, t = {}, diff = {}'
              .format(target_label, y_pred[:n_eval, target_label],
                      t[:n_eval, target_label], diff))

    # --- evaluate ---
    # To calc loss/accuracy, we can use `Evaluator`, `ROCAUCEvaluator`
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(
        test_iterator, regressor, converter=concat_mols, device=args.gpu)()
    print('Evaluation result: ', eval_result)

    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
