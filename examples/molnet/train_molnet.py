import argparse
import os

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E
import numpy

from chainer_chemistry import datasets as D
from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet, RSGCN  # NOQA
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.training.extensions import ROCAUCEvaluator


class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp=None):
        """Graph Convolution Predictor

        It sequentially combines graph convolution network and multi layer
        perceptron.
        `graph_conv` gathers the each node's feature to extract graph feature,
        `mlp` processed the extracted graph feature to calculate final output.

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer.
                It can be `None` if no operation is necessary after
                `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        if self.mlp:
            x = self.mlp(x)
        return x


def main():
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    dataset_names = list(molnet_default_config.keys())

    parser = argparse.ArgumentParser(description='molnet example')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        default='nfp')
    parser.add_argument('--label', '-l', type=str, default='',
                        help='target label for regression, empty string means '
                        'to predict all property at once')
    parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--dataset', '-d', type=str, choices=dataset_names,
                        default='bbbp')
    parser.add_argument('--protocol', type=int, default=2)
    parser.add_argument('--model-filename', type=str, default='regressor.pkl')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='Number of data to be parsed from parser.'
                             '-1 indicates to parse all data.')
    args = parser.parse_args()
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data
    n_unit = args.unit_num
    conv_layers = args.conv_layers
    print('Use {} dataset'.format(dataset_name))

    if args.label:
        labels = args.label
        cache_dir = os.path.join('input', '{}_{}_{}'.format(dataset_name,
                                                            method, labels))
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_{}_all'.format(dataset_name,
                                                             method))
        class_num = len(molnet_default_config[args.dataset]['tasks'])

    # Dataset preparation
    def get_dataset_paths(cache_dir, num_data):
        filepaths = []
        for filetype in ['train', 'valid', 'test']:
            filename = filetype+'_data'
            if num_data >= 0:
                filename += '_' + str(num_data)
            filename += '.npz'
            filepath = os.path.join(cache_dir, filename)
            filepaths.append(filepath)
        return filepaths
    filepaths = get_dataset_paths(cache_dir, num_data)
    if all([os.path.exists(fpath) for fpath in filepaths]):
        datasets = []
        for fpath in filepaths:
            print('load from cache {}'.format(fpath))
            datasets.append(NumpyTupleDataset.load(fpath))
    else:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        # only use first 100 for debug if num_data >= 0
        target_index = numpy.arange(num_data) if num_data >= 0 else None
        datasets = D.molnet.get_molnet_dataset(dataset_name, preprocessor,
                                               labels=labels,
                                               target_index=target_index)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        datasets = datasets['dataset']
        for i, fpath in enumerate(filepaths):
            NumpyTupleDataset.save(fpath, datasets[i])

    train, val, _ = datasets

    # Network
    if method == 'nfp':
        print('Train NFP model...')
        predictor = GraphConvPredictor(NFP(out_dim=n_unit, hidden_dim=n_unit,
                                       n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Train GGNN model...')
        predictor = GraphConvPredictor(GGNN(out_dim=n_unit, hidden_dim=n_unit,
                                        n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Train SchNet model...')
        predictor = GraphConvPredictor(
            SchNet(out_dim=class_num, hidden_dim=n_unit, n_layers=conv_layers),
            None)
    elif method == 'weavenet':
        print('Train WeaveNet model...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        predictor = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'rsgcn':
        print('Train RSGCN model...')
        predictor = GraphConvPredictor(
            RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid method {}'.format(method))

    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(val, args.batchsize,
                                        repeat=False, shuffle=False)

    metrics_fun = molnet_default_config[dataset_name]['metrics']
    loss_fun = molnet_default_config[dataset_name]['loss']
    task_type = molnet_default_config[dataset_name]['task_type']
    if task_type == 'regression':
        model = Regressor(predictor, lossfun=loss_fun, metrics_fun=metrics_fun,
                          device=args.gpu)
        # TODO(nakago): Use standard scaler for regression task
    elif task_type == 'classification':
        model = Classifier(predictor, lossfun=loss_fun,
                           metrics_fun=metrics_fun, device=args.gpu)
    else:
        raise NotImplementedError(
            'Not implemented task_type = {}'.format(task_type))

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       converter=concat_mols)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(E.Evaluator(val_iter, model, device=args.gpu,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())
    print_report_targets = ['epoch', 'main/loss', 'validation/main/loss']
    if metrics_fun is not None and type(metrics_fun) == dict:
        for m_k in metrics_fun.keys():
            print_report_targets.append('main/'+m_k)
            print_report_targets.append('validation/main/'+m_k)
    if task_type == 'classification':
        # Evaluation for train data takes time, skip for now.
        # trainer.extend(ROCAUCEvaluator(
        #     train_iter, model, device=args.gpu, eval_func=predictor,
        #     converter=concat_mols, name='train', raise_value_error=False))
        # print_report_targets.append('train/main/roc_auc')
        trainer.extend(ROCAUCEvaluator(
            val_iter, model, device=args.gpu, eval_func=predictor,
            converter=concat_mols, name='val', raise_value_error=False))
        print_report_targets.append('val/main/roc_auc')
    print_report_targets.append('elapsed_time')
    trainer.extend(E.PrintReport(print_report_targets))
    trainer.extend(E.ProgressBar())
    trainer.run()

    # --- save model ---
    protocol = args.protocol
    model.save_pickle(os.path.join(args.out, args.model_filename),
                      protocol=protocol)


if __name__ == '__main__':
    main()
