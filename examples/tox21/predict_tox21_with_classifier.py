import os

import argparse
import json

import chainer
import numpy

from chainer import cuda
import chainer.functions as F
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
from rdkit import RDLogger
import six

from chainer_chemistry.dataset.converters import converter_method_dict
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator  # NOQA

import data


# Disable errors by RDKit occurred in preprocessing Tox21 dataset.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def main():
    parser = argparse.ArgumentParser(
        description='Predict with a trained model.')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='Path to the result directory of the training '
                        'script.')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='batch size')
    parser.add_argument(
        '--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='file name for pickled model')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='Number of data to be parsed from parser.'
                             '-1 indicates to parse all data.')
    args = parser.parse_args()

    with open(os.path.join(args.in_dir, 'config.json'), 'r') as i:
        config = json.loads(i.read())

    method = config['method']
    labels = config['labels']

    _, test, _ = data.load_dataset(method, labels, num_data=args.num_data)
    y_test = test.get_datasets()[-1]

    device = chainer.get_device(args.device)
    # Load pretrained model
    clf = Classifier.load_pickle(
        os.path.join(args.in_dir, args.model_filename),
        device=device)  # type: Classifier

    # ---- predict ---
    print('Predicting...')

    # We need to feed only input features `x` to `predict`/`predict_proba`.
    # This converter extracts only inputs (x1, x2, ...) from the features which
    # consist of input `x` and label `t` (x1, x2, ..., t).
    converter = converter_method_dict[method]

    def extract_inputs(batch, device=None):
        return converter(batch, device=device)[:-1]

    def postprocess_pred(x):
        x_array = cuda.to_cpu(x.data)
        return numpy.where(x_array > 0, 1, 0)
    y_pred = clf.predict(test, converter=extract_inputs,
                         postprocess_fn=postprocess_pred)
    y_proba = clf.predict_proba(test, converter=extract_inputs,
                                postprocess_fn=F.sigmoid)

    # `predict` method returns the prediction label (0: non-toxic, 1:toxic)
    print('y_pread.shape = {}, y_pred[:5, 0] = {}'
          .format(y_pred.shape, y_pred[:5, 0]))
    # `predict_proba` method returns the probability to be toxic
    print('y_proba.shape = {}, y_proba[:5, 0] = {}'
          .format(y_proba.shape, y_proba[:5, 0]))
    # --- predict end ---

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    if y_pred.shape != y_test.shape:
        raise RuntimeError('The shape of the prediction result array and '
                           'that of the ground truth array do not match. '
                           'Contents of the input directory may be corrupted '
                           'or modified.')

    statistics = []
    for t, p in six.moves.zip(y_test.T, y_pred.T):
        idx = t != -1
        n_correct = (t[idx] == p[idx]).sum()
        n_total = len(t[idx])
        accuracy = float(n_correct) / n_total
        statistics.append([n_correct, n_total, accuracy])

    print('{:>6} {:>8} {:>8} {:>8}'
          .format('TaskID', 'Correct', 'Total', 'Accuracy'))
    for idx, (n_correct, n_total, accuracy) in enumerate(statistics):
        print('task{:>2} {:>8} {:>8} {:>8.4f}'
              .format(idx, n_correct, n_total, accuracy))

    prediction_result_file = 'prediction.npz'
    print('Save prediction result to {}'.format(prediction_result_file))
    numpy.savez_compressed(prediction_result_file, y_pred)

    # --- evaluate ---
    # To calc loss/accuracy, we can use `Evaluator`, `ROCAUCEvaluator`
    print('Evaluating...')
    test_iterator = SerialIterator(test, 16, repeat=False, shuffle=False)
    eval_result = Evaluator(
        test_iterator, clf, converter=converter, device=device)()
    print('Evaluation result: ', eval_result)
    rocauc_result = ROCAUCEvaluator(
        test_iterator, clf, converter=converter, device=device,
        eval_func=clf.predictor, name='test', ignore_labels=-1)()
    print('ROCAUC Evaluation result: ', rocauc_result)
    with open(os.path.join(args.in_dir, 'eval_result.json'), 'w') as f:
        json.dump(rocauc_result, f)
    # --- evaluate end ---


if __name__ == '__main__':
    main()
