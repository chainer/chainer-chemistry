#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import numpy
import os
import pandas
import pickle

from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

from chainer import cuda
from chainer.datasets import split_dataset_random
from chainer import Variable

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models.prediction import Regressor

# These import is necessary for pickle to work
from sklearn.preprocessing import StandardScaler  # NOQA
from train_qm9 import GraphConvPredictor  # NOQA
from train_qm9 import MeanAbsError, RootMeanSqrError  # NOQA


class ScaledGraphConvPredictor(GraphConvPredictor):
    def __init__(self, *args, **kwargs):
        """Initializes the (scaled) graph convolution predictor. This uses
        a standard scaler to rescale the predicted labels.
        """
        super(ScaledGraphConvPredictor, self).__init__(*args, **kwargs)

    def __call__(self, atoms, adjs):
        h = super(ScaledGraphConvPredictor, self).__call__(atoms, adjs)
        scaler_available = hasattr(self, 'scaler')
        numpy_data = isinstance(h.data, numpy.ndarray)

        if scaler_available:
            h = self.scaler.inverse_transform(cuda.to_cpu(h.data))
            if not numpy_data:
                h = cuda.to_gpu(h)
        return Variable(h)


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'gat']
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
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='id of gpu to use; negative value means running'
                        'the code on cpu')
    parser.add_argument('--seed', '-s', type=int, default=777,
                        help='random seed value')
    parser.add_argument('--train-data-ratio', '-r', type=float, default=0.7,
                        help='ratio of training data w.r.t the dataset')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory to load model data from')
    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    method = args.method
    if args.label:
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
    else:
        labels = D.get_qm9_label_names()
        cache_dir = os.path.join('input', '{}_all'.format(method))

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
        print('Loading cached data from {}.'.format(dataset_cache_path))
        dataset = NumpyTupleDataset.load(dataset_cache_path)
    if dataset is None:
        print('Preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        dataset = D.get_qm9(preprocessor, labels=labels)

        # Cache the newly preprocessed dataset.
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        NumpyTupleDataset.save(dataset_cache_path, dataset)

    # Load the standard scaler parameters, if necessary.
    if args.scale == 'standardize':
        scaler_path = os.path.join(args.in_dir, 'scaler.pkl')
        print('Loading scaler parameters from {}.'.format(scaler_path))
        with open(scaler_path, mode='rb') as f:
            scaler = pickle.load(f)
    else:
        print('No standard scaling was selected.')
        scaler = None

    # Split the dataset into training and testing.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    _, test = split_dataset_random(dataset, train_data_size, args.seed)

    # Use a predictor with scaled output labels.
    model_path = os.path.join(args.in_dir, args.model_filename)
    regressor = Regressor.load_pickle(model_path, device=args.gpu)

    # Replace the default predictor with one that scales the output labels.
    scaled_predictor = ScaledGraphConvPredictor(regressor.predictor)
    scaled_predictor.scaler = scaler
    regressor.predictor = scaled_predictor

    # This callback function extracts only the inputs and discards the labels.
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]

    # Predict the output labels.
    print('Predicting...')
    y_pred = regressor.predict(test, converter=extract_inputs)

    # Extract the ground-truth labels.
    t = concat_mols(test, device=-1)[-1]
    n_eval = 10

    # Construct dataframe.
    df_dict = {}
    for i, l in enumerate(labels):
        df_dict.update({'y_pred_{}'.format(l): y_pred[:, i],
                        't_{}'.format(l): t[:, i], })
    df = pandas.DataFrame(df_dict)

    # Show a prediction/ground truth table with 5 random examples.
    print(df.sample(5))
    for target_label in range(y_pred.shape[1]):
        diff = y_pred[:n_eval, target_label] - t[:n_eval, target_label]
        print('target_label = {}, y_pred = {}, t = {}, diff = {}'
              .format(target_label, y_pred[:n_eval, target_label],
                      t[:n_eval, target_label], diff))

    # Run an evaluator on the test dataset.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                            device=args.gpu)()
    print('Evaluation result: ', eval_result)

    # Save the evaluation results.
    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
