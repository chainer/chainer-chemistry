import os

import argparse
import chainer
import glob
import json
import numpy
from chainer import cuda
from chainer.iterators import SerialIterator
from chainer.training.extensions import Evaluator
import chainer.functions as F
from rdkit import RDLogger
import six

from chainer_chemistry import datasets as D
from chainer_chemistry.dataset.converters import concat_mols
try:
    from chainer_chemistry.models.prediction import Classifier
except ImportError:
    print('[WARNING] If you want to use Classifier in Chainer Chemistry, '
          'please install the library from master branch.\n See '
          'https://github.com/pfnet-research/chainer-chemistry#installation'
          ' for detail.')
from chainer_chemistry.training.extensions.roc_auc_evaluator import \
    ROCAUCEvaluator

import data
import predictor


# Disable errors by RDKit occurred in preprocessing Tox21 dataset.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _find_latest_snapshot(in_dir, prefix='snapshot_iter_'):
    snapshot_files = glob.glob(os.path.join(in_dir, prefix + '*'))
    ret = None
    max_iteration_number = -1
    for filepath in snapshot_files:
        try:
            basename = os.path.basename(filepath)
            iteration_number = int(basename.lstrip(prefix))
            if max_iteration_number < iteration_number:
                ret = filepath
                max_iteration_number = iteration_number
        except Exception:
            continue
    if ret is None:
        raise ValueError('No snapshot files found in {}'.format(in_dir))
    return ret


def main():
    label_names = D.get_tox21_label_names()

    parser = argparse.ArgumentParser(
        description='Predict with a trained model.')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='Path to the result directory of the training '
                        'script.')
    parser.add_argument('--trainer-snapshot', '-s', type=str, default='',
                        help='Path to the snapshot file of the Chainer '
                        'trainer from which serialized model parameters '
                        'are extracted. If it is not specified, this '
                        'script searches the training result directory '
                        'for the latest snapshot, assuming that '
                        'the naming convension of snapshot files is '
                        '`snapshot_iter_N` where N is the number of '
                        'iterations, which is the default configuration '
                        'of Chainer.')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                        'not to use GPU and to run the code in CPU.')
    args = parser.parse_args()

    with open(os.path.join(args.in_dir, 'config.json'), 'r') as i:
        config = json.loads(i.read())

    method = config['method']
    labels = config['labels']
    if labels:
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        class_num = len(label_names)

    _, test, _ = data.load_dataset(method, labels)
    y_test = test.get_datasets()[-1]

    # Load pretrained model
    predictor_ = predictor.build_predictor(
        method, config['unit_num'], config['conv_layers'], class_num)
    snapshot_file = args.trainer_snapshot
    if not snapshot_file:
        snapshot_file = _find_latest_snapshot(args.in_dir)
    print('Loading pretrained model parameters from {}'.format(snapshot_file))
    chainer.serializers.load_npz(snapshot_file,
                                 predictor_, 'updater/model:main/predictor/')

    clf = Classifier(predictor=predictor_, device=args.gpu,
                     lossfun=F.sigmoid_cross_entropy,
                     metrics_fun=F.binary_accuracy)

    # ---- predict ---
    print('Predicting...')

    # We need to feed only input features `x` to `predict`/`predict_proba`.
    # This converter extracts only inputs (x1, x2, ...) from the features which
    # consist of input `x` and label `t` (x1, x2, ..., t).
    def extract_inputs(batch, device=None):
        return concat_mols(batch, device=device)[:-1]

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
        test_iterator, clf, converter=concat_mols, device=args.gpu)()
    print('Evaluation result: ', eval_result)
    rocauc_result = ROCAUCEvaluator(
        test_iterator, clf, converter=concat_mols, device=args.gpu,
        eval_func=predictor_, name='test', ignore_labels=-1)()
    print('ROCAUC Evaluation result: ', rocauc_result)
    with open('result.json', 'w') as f:
        json.dump(rocauc_result, f)
    # --- evaluate end ---


if __name__ == '__main__':
    main()
