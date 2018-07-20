#!/usr/bin/env python

from __future__ import print_function

import chainer
import json
import numpy
import os
import pickle

from argparse import ArgumentParser
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

from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset

# These import is necessary for pickle to work
from sklearn.preprocessing import StandardScaler  # NOQA
from train_custom_dataset import GraphConvPredictor  # NOQA
from train_custom_dataset import ScaledAbsError  # NOQA


def parse_arguments():
    # Lists of supported preprocessing methods/models.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on a custom dataset')
    parser.add_argument('--datafile', '-d', type=str,
                        default='dataset_test.csv',
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
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='directory containing the saved model')
    parser.add_argument('--model-filename', type=str, default='regressor.pkl',
                        help='saved model filename')
    return parser.parse_args()


def main():
    # Parse the arguments.
    args = parse_arguments()

    if args.label:
        labels = args.label
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        sys.exit("Error: No target label was specified.")

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

    test = dataset

    # Set up the regressor.
    model_path = os.path.join(args.in_dir, args.model_filename)
    regressor = Regressor.load_pickle(model_path, device=args.gpu)

    # Extracts only inputs (x1, x2, ...) from the features which consist of
    # input `x` and label `t` (x1, x2, ..., t).
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]

    def postprocess_fn(x):
        if ss is not None:
            # Rescale model's output.
            if isinstance(x, Variable):
                x = x.data
                scaled_x = ss.inverse_transform(cuda.to_cpu(x))
                return scaled_x
        else:
            return x

    # Perform the prediction.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                            device=args.gpu)()
    print('Evaluation result: ', eval_result)

    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
