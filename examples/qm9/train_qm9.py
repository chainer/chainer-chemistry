#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print('You need to install scikit-learn to run this example,'
          'please run `pip install -U scikit-learn`')

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass


import chainer
from chainer import functions as F, cuda, Variable
from chainer import iterators as I
from chainer import links as L
from chainer import optimizers as O
from chainer import training
from chainer.datasets import split_dataset_random
from chainer.training import extensions as E
import numpy

from chainerchem import datasets as D
from chainerchem.models import MLP, NFP, GGNN, SchNet, WeaveNet
from chainerchem.dataset.converters import concat_mols
from chainerchem.dataset.preprocessors import preprocess_method_dict
from chainerchem.datasets import NumpyTupleDataset


class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp):
        """
        
        Args:
            graph_conv: graph convolution network to obtain molecule feature 
                        representation
            mlp: multi layer perceptron, used as final connected layer
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        x = self.mlp(x)
        return x

    def _predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)

    def predict(self, *args, batchsize=32, device=-1):
        if device >= 0:
            chainer.cuda.get_device_from_id(device).use()
            self.to_gpu()  # Copy the model to the GPU

        # TODO: Not test yet, check behavior
        data = args[0]
        y_list = []
        for i in range(0, len(data), batchsize):
            atoms, adjs = concat_mols(data[i:i + batchsize], device=device)[:2]
            y = self._predict(atoms, adjs)
            y_list.append(cuda.to_cpu(y.data))
        y_array = numpy.concatenate(y_list, axis=0)
        return y_array


def main():
    # Supported preprocessing/network list
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet']
    label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                   'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    parser = argparse.ArgumentParser(
        description='Regression with QM9.')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp')
    parser.add_argument('--label', '-l', type=str, choices=label_names,
                        default='', help='target label for regression, '
                                         'empty string means to predict all '
                                         'property at once')
    parser.add_argument('--scale', type=str, default='standardize',
                        help='Label scaling method')
    parser.add_argument('--conv_layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--unit_num', '-u', type=int, default=16)
    args = parser.parse_args()

    seed = 777
    train_data_ratio = 0.7
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

    #cache_dir = os.path.join('input', '{}'.format(method))
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
        ss = StandardScaler()
        labels = ss.fit_transform(dataset.get_datasets()[-1])
        dataset = NumpyTupleDataset(*dataset.get_datasets()[:-1], labels)

    train_data_size = int(len(dataset) * train_data_ratio)
    train, val = split_dataset_random(dataset, train_data_size, seed)

    # Network
    if method == 'nfp':
        print('Train NFP model...')
        n_unit = args.unit_num
        model = GraphConvPredictor(NFP(out_dim=n_unit, hidden_dim=n_unit,
                                       n_layers=args.conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Train GGNN model...')
        n_unit = args.unit_num
        model = GraphConvPredictor(GGNN(out_dim=n_unit, hidden_dim=n_unit,
                                        n_layers=args.conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Train SchNet model...')
        model = SchNet(out_dim=class_num)
    elif method == 'weavenet':
        print('Train WeaveNet model...')
        n_unit = args.unit_num
        n_atom = 20
        n_layer = args.conv_layers
        # n_layer = 1
        n_sub_layer = 1
        weave_channels = [50] * n_layer
        model = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))

    else:
        print('[ERROR] Invalid mode')
        exit()

    train_iter = I.SerialIterator(train, args.batchsize)
    val_iter = I.SerialIterator(val, args.batchsize,
                                repeat=False, shuffle=False)

    def scaled_abs_error(x0, x1):
        if isinstance(x0, Variable):
            x0 = cuda.to_cpu(x0.data)
        if isinstance(x1, Variable):
            x1 = cuda.to_cpu(x1.data)
        scaled_x0 = ss.inverse_transform(cuda.to_cpu(x0))
        scaled_x1 = ss.inverse_transform(cuda.to_cpu(x1))
        return numpy.mean(numpy.absolute(scaled_x0 - scaled_x1), axis=0)[0]

    classifier = L.Classifier(model, lossfun=F.mean_squared_error,
                              accfun=scaled_abs_error)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(classifier)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(val_iter, classifier, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                  'validation/main/loss',
                                  'validation/main/accuracy',
                                  'elapsed_time']))
    trainer.extend(E.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()
