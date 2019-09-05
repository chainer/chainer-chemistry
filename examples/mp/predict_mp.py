#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import numpy
import pandas

import chainer
from chainer import cuda
from chainer import functions as F
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from chainer.datasets import split_dataset_random

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass


from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.utils import save_json
from chainer_chemistry.dataset.preprocessors.mp_megnet_preprocessor import MPMEGNetPreprocessor
from chainer_chemistry.datasets.mp import MPDataset


def rmse(x0, x1):
    return F.sqrt(F.mean_squared_error(x0, x1))


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    label_names = ['formation_energy_per_atom', 'energy', 'band_gap', 'efermi',
                   'K_VRH', 'G_VRH', 'poisson_ratio']
    method_list = ['megnet']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = argparse.ArgumentParser(description='Regression on Material Project Data.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='megnet')
    parser.add_argument('--label', '-l', type=str, choices=label_names, 
                        default='formation_energy_per_atom',
                        help='target label for regression')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
    parser.add_argument(
        '--device', '-d', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
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
    # TODO : 今後不要になる
    parser.add_argument("--data_dir", "-dd", type=str, default="",
                        help="path to data dir")
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()
    device = chainer.get_device(args.device)

    # Set up some useful variables that will be used later on.
    method = args.method
    labels = args.label
    target_list = [labels]
    class_num = len(labels) if isinstance(labels, list) else 1
    cache_dir = os.path.join('input', '{}_{}'.format(method, labels))

    # Get the filename corresponding to the cached dataset, based on the amount
    # of data samples that need to be parsed from the original dataset.
    num_data = args.num_data
    if num_data >= 0:
        dataset_filename = 'data_{}.npz'.format(num_data)
    else:
        dataset_filename = 'data.npz'

    # Load the cached dataset.
    dataset = MPDataset()
    dataset_cache_path = os.path.join(cache_dir, dataset_filename)
    result = dataset.load_pickle(dataset_cache_path)

    # load datasets from Material Project Database
    if result is False:
        preprocessor = MPMEGNetPreprocessor()
        dataset.get_mp(args.data_dir, target_list, preprocessor, args.num_data)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        dataset.save_pickle(dataset_cache_path)

    # Use a predictor with scaled output labels.
    model_path = os.path.join(args.in_dir, args.model_filename)
    regressor = Regressor.load_pickle(model_path, device=device)

    # Split the dataset into training and testing.
    train_data_size = int(len(dataset) * args.train_data_ratio)
    _, test = split_dataset_random(dataset, train_data_size, args.seed)

   # This callback function extracts only the inputs and discards the labels.
    @chainer.dataset.converter()
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]

    # Predict the output labels.
    print('Predicting...')
    y_pred = regressor.predict(
        test, converter=extract_inputs)

    # Extract the ground-truth labels as numpy array.
    original_t = concat_mols(test, device=-1)[-1]

    # Construct dataframe.
    df_dict = {}
    for i, l in enumerate(target_list):
        df_dict.update({'y_pred_{}'.format(l): y_pred[:, i],
                        't_{}'.format(l): original_t[:, i], })
    df = pandas.DataFrame(df_dict)

    # Show a prediction/ground truth table with 5 random examples.
    print(df.sample(5))

    n_eval = 10
    for target_label in range(y_pred.shape[1]):
        label_name = labels[target_label]
        diff = y_pred[:n_eval, target_label] - original_t[:n_eval,
                                                          target_label]
        print('label_name = {}, y_pred = {}, t = {}, diff = {}'
              .format(label_name, y_pred[:n_eval, target_label],
                      original_t[:n_eval, target_label], diff))

    # Run an evaluator on the test dataset.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                            device=device)()
    print('Evaluation result: ', eval_result)
    # Save the evaluation results.
    save_json(os.path.join(args.in_dir, 'eval_result.json'), eval_result)

    # Calculate mean abs error for each label
    mae = numpy.mean(numpy.abs(y_pred - original_t), axis=0)
    eval_result = {}
    for i, l in enumerate(target_list):
        eval_result.update({l: mae[i]})
    save_json(os.path.join(args.in_dir, 'eval_result_mae.json'), eval_result)


if __name__ == '__main__':
    main()
