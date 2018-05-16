import os

import argparse
import chainer
import glob
import json
import numpy
from rdkit import RDLogger
import six

from chainer_chemistry import datasets as D

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
        description='Inference with a trained model.')
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
    parser.add_argument('--num-data', type=int, default=-1,
                        help='Number of data to be parsed from parser.'
                             '-1 indicates to parse all data.')
    args = parser.parse_args()

    with open(os.path.join(args.in_dir, 'config.json'), 'r') as i:
        config = json.loads(i.read())

    method = config['method']
    labels = config['labels']
    if labels:
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        class_num = len(label_names)

    _, test, _ = data.load_dataset(method, labels, num_data=args.num_data)
    test = test.get_datasets()
    X_test = D.NumpyTupleDataset(*test[:-1])
    y_test = test[-1]

    # Load pretrained model
    predictor_ = predictor.build_predictor(
        method, config['unit_num'], config['conv_layers'], class_num)
    snapshot_file = args.trainer_snapshot
    if not snapshot_file:
        snapshot_file = _find_latest_snapshot(args.in_dir)
    print('Loading pretrained model parameters from {}'.format(snapshot_file))
    chainer.serializers.load_npz(snapshot_file,
                                 predictor_, 'updater/model:main/predictor/',
                                 strict=False)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        predictor_.to_gpu()

    inference_loop = predictor.InferenceLoop(predictor_)
    y_pred = inference_loop.inference(X_test)
    if y_pred.shape != y_test.shape:
        raise RuntimeError('The shape of the prediction result array and '
                           'that of the ground truth array do not match. '
                           'Contents of the input directory may be corrupted '
                           'or modified.')

    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_test = y_test[:, None]

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


if __name__ == '__main__':
    main()
