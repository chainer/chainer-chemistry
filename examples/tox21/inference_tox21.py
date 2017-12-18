import os

import argparse
import chainer
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


def main():
    label_names = D.get_tox21_label_names()

    parser = argparse.ArgumentParser(
        description='Inference with a trained model.')
    parser.add_argument('--in-dir', '-i', type=str, default='result',
                        help='path result directory of training')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                        'not to use GPU and to run the code in CPU.')
    args = parser.parse_args()

    with open(os.path.join(args.in_dir, 'config.json'), 'r') as i:
        config = json.loads(i.read())

    method = config['method']
    if method == 'schnet':
        raise ValueError('Currently SchNet does not support prediction.')

    labels = config['labels']
    if labels:
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        class_num = len(label_names)

    _, test, _ = data.load_dataset(method, labels)
    test = test.get_datasets()
    X_test = D.NumpyTupleDataset(*test[:-1])
    y_test = test[-1]

    predictor_ = predictor.build_predictor(
        method, config['unit_num'], config['conv_layers'], class_num)
    chainer.serializers.load_npz(os.path.join(args.in_dir, 'snapshot_iter_92'),
                                 predictor_, 'updater/model:main/predictor/')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        predictor_.to_gpu()

    inference_loop = predictor.InferenceLoop(predictor_)
    y_pred = inference_loop.inference(X_test)

    for t, p in six.moves.zip(y_test.T, y_pred.T):
        idx = t != -1
        n_correct = (t[idx] == p[idx]).sum()
        n_total = len(t[idx])
        print(float(n_correct) / n_total)

    numpy.save('prediction.npy', y_pred)

if __name__ == '__main__':
    main()
