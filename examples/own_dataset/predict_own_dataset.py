#!/usr/bin/env python

from __future__ import print_function

import json
import numpy
import os
import pickle

from argparse import ArgumentParser
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator

from chainer import cuda
from chainer import Variable

from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.parsers import CSVFileParser
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict

# These imports are necessary for pickle to work.
from sklearn.preprocessing import StandardScaler  # NOQA
from train_own_dataset import GraphConvPredictor  # NOQA
from train_own_dataset import MeanAbsError, RootMeanSqrError  # NOQA


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
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    scale_list = ['standardize', 'none']

    # Set up the argument parser.
    parser = ArgumentParser(description='Regression on own dataset')
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
    else:
        raise ValueError('No target label was specified.')

    # Dataset preparation.
    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)

    print('Preprocessing dataset...')
    preprocessor = preprocess_method_dict[args.method]()
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES')
    dataset = parser.parse(args.datafile)['dataset']

    # Load the standard scaler parameters, if necessary.
    if args.scale == 'standardize':
        with open(os.path.join(args.in_dir, 'scaler.pkl'), mode='rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None
    test = dataset

    print('Predicting...')
    # Set up the regressor.
    model_path = os.path.join(args.in_dir, args.model_filename)
    regressor = Regressor.load_pickle(model_path, device=args.gpu)
    scaled_predictor = ScaledGraphConvPredictor(regressor.predictor)
    scaled_predictor.scaler = scaler
    regressor.predictor = scaled_predictor

    # Perform the prediction.
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(test_iterator, regressor, converter=concat_mols,
                            device=args.gpu)()

    # Prevents the loss function from becoming a cupy.core.core.ndarray object
    # when using the GPU. This hack will be removed as soon as the cause of
    # the issue is found and properly fixed.
    loss = numpy.asscalar(cuda.to_cpu(eval_result['main/loss']))
    eval_result['main/loss'] = loss
    print('Evaluation result: ', eval_result)

    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
