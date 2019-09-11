#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy
import os
import types

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training

from chainer.training import extensions as E
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.links import StandardScaler
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions import BatchEvaluator, ROCAUCEvaluator  # NOQA
from chainer_chemistry.training.extensions.auto_print_report import AutoPrintReport  # NOQA


def parse_arguments():
    # Lists of supported preprocessing methods/models and datasets.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'gin', 'gnnfilm',
                   'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm']
    dataset_names = list(molnet_default_config.keys())
    scale_list = ['standardize', 'none']

    parser = argparse.ArgumentParser(description='molnet example')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', type=str, default='',
                        help='target label for regression; empty string means '
                        'predicting all properties at once')
    parser.add_argument('--conv-layers', '-c', type=int, default=4,
                        help='number of convolution layers')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='batch size')
    parser.add_argument(
        '--device', type=str, default='-1',
        help='Device specifier. Either ChainerX device specifier or an '
             'integer. If non-negative integer, CuPy arrays with specified '
             'device id are used. If negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to save the computed model to')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--unit-num', '-u', type=int, default=16,
                        help='number of units in one layer of the model')
    parser.add_argument('--dataset', '-d', type=str, choices=dataset_names,
                        default='bbbp',
                        help='name of the dataset that training is run on')
    parser.add_argument('--protocol', type=int, default=2,
                        help='pickle protocol version')
    parser.add_argument('--num-data', type=int, default=-1,
                        help='amount of data to be parsed; -1 indicates '
                        'parsing all data.')
    parser.add_argument('--scale', type=str, choices=scale_list,
                        help='label scaling method', default='standardize')
    return parser.parse_args()


def dataset_part_filename(dataset_part, num_data):
    """Returns the filename corresponding to a train/valid/test parts of a
    dataset, based on the amount of data samples that need to be parsed.
    Args:
        dataset_part: String containing any of the following 'train', 'valid'
                      or 'test'.
        num_data: Amount of data samples to be parsed from the dataset.
    """
    if num_data >= 0:
        return '{}_data_{}.npz'.format(dataset_part, str(num_data))
    return '{}_data.npz'.format(dataset_part)


def download_entire_dataset(dataset_name, num_data, labels, method, cache_dir):
    """Downloads the train/valid/test parts of a dataset and stores them in the
    cache directory.
    Args:
        dataset_name: Dataset to be downloaded.
        num_data: Amount of data samples to be parsed from the dataset.
        labels: Target labels for regression.
        method: Method name. See `parse_arguments`.
        cache_dir: Directory to store the dataset to.
    """

    print('Downloading {}...'.format(dataset_name))
    preprocessor = preprocess_method_dict[method]()

    # Select the first `num_data` samples from the dataset.
    target_index = numpy.arange(num_data) if num_data >= 0 else None
    dataset_parts = D.molnet.get_molnet_dataset(dataset_name, preprocessor,
                                                labels=labels,
                                                target_index=target_index)
    dataset_parts = dataset_parts['dataset']

    # Cache the downloaded dataset.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for i, part in enumerate(['train', 'valid', 'test']):
        filename = dataset_part_filename(part, num_data)
        path = os.path.join(cache_dir, filename)
        NumpyTupleDataset.save(path, dataset_parts[i])
    return dataset_parts


def fit_scaler(datasets):
    """Standardizes (scales) the dataset labels.
    Args:
        datasets: Tuple containing the datasets.
    Returns:
        Datasets with standardized labels and the scaler object.
    """
    scaler = StandardScaler()

    # Collect all labels in order to apply scaling over the entire dataset.
    labels = None
    offsets = []
    for dataset in datasets:
        if labels is None:
            labels = dataset.get_datasets()[-1]
        else:
            labels = numpy.vstack([labels, dataset.get_datasets()[-1]])
        offsets.append(len(labels))

    scaler.fit(labels)

    return scaler


def main():
    args = parse_arguments()

    # Set up some useful variables that will be used later on.
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data
    n_unit = args.unit_num
    conv_layers = args.conv_layers

    task_type = molnet_default_config[dataset_name]['task_type']
    model_filename = {'classification': 'classifier.pkl',
                      'regression': 'regressor.pkl'}

    print('Using dataset: {}...'.format(dataset_name))

    # Set up some useful variables that will be used later on.
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

    # Load the train and validation parts of the dataset.
    filenames = [dataset_part_filename(p, num_data)
                 for p in ['train', 'valid']]

    paths = [os.path.join(cache_dir, f) for f in filenames]
    if all([os.path.exists(path) for path in paths]):
        dataset_parts = []
        for path in paths:
            print('Loading cached dataset from {}.'.format(path))
            dataset_parts.append(NumpyTupleDataset.load(path))
    else:
        dataset_parts = download_entire_dataset(dataset_name, num_data, labels,
                                                method, cache_dir)
    train, valid = dataset_parts[0], dataset_parts[1]

    # Scale the label values, if necessary.
    scaler = None
    if args.scale == 'standardize':
        if task_type == 'regression':
            print('Applying standard scaling to the labels.')
            scaler = fit_scaler(dataset_parts)
        else:
            print('Label scaling is not available for classification tasks.')
    else:
        print('No label scaling was selected.')

    # Set up the predictor.
    predictor = set_up_predictor(method, n_unit, conv_layers, class_num,
                                 label_scaler=scaler)

    # Set up the iterators.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False,
                                          shuffle=False)

    # Load metrics for the current dataset.
    metrics = molnet_default_config[dataset_name]['metrics']
    metrics_fun = {k: v for k, v in metrics.items()
                   if isinstance(v, types.FunctionType)}
    loss_fun = molnet_default_config[dataset_name]['loss']

    device = chainer.get_device(args.device)
    if task_type == 'regression':
        model = Regressor(predictor, lossfun=loss_fun,
                          metrics_fun=metrics_fun, device=device)
    elif task_type == 'classification':
        model = Classifier(predictor, lossfun=loss_fun,
                           metrics_fun=metrics_fun, device=device)
    else:
        raise ValueError('Invalid task type ({}) encountered when processing '
                         'dataset ({}).'.format(task_type, dataset_name))

    # Set up the optimizer.
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Save model-related output to this directory.
    model_dir = os.path.join(args.out, os.path.basename(cache_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Set up the updater.
    updater = training.StandardUpdater(train_iter, optimizer, device=device,
                                       converter=concat_mols)

    # Set up the trainer.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=model_dir)
    trainer.extend(E.Evaluator(valid_iter, model, device=device,
                               converter=concat_mols))
    trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(E.LogReport())

    # TODO: consider go/no-go of the following block
    # # (i) more reporting for val/evalutaion
    # # (ii) best validation score snapshot
    # if task_type == 'regression':
    #     metric_name_list = list(metrics.keys())
    #     if 'RMSE' in metric_name_list:
    #         trainer.extend(E.snapshot_object(model, "best_val_" + model_filename[task_type]),
    #                        trigger=training.triggers.MinValueTrigger('validation/main/RMSE'))
    #     elif 'MAE' in metric_name_list:
    #         trainer.extend(E.snapshot_object(model, "best_val_" + model_filename[task_type]),
    #                        trigger=training.triggers.MinValueTrigger('validation/main/MAE'))
    #     else:
    #         print("[WARNING] No validation metric defined?")
    #
    # elif task_type == 'classification':
    #     train_eval_iter = iterators.SerialIterator(
    #         train, args.batchsize, repeat=False, shuffle=False)
    #     trainer.extend(ROCAUCEvaluator(
    #         train_eval_iter, predictor, eval_func=predictor,
    #         device=args.gpu, converter=concat_mols, name='train',
    #         pos_labels=1, ignore_labels=-1, raise_value_error=False))
    #     # extension name='validation' is already used by `Evaluator`,
    #     # instead extension name `val` is used.
    #     trainer.extend(ROCAUCEvaluator(
    #         valid_iter, predictor, eval_func=predictor,
    #         device=args.gpu, converter=concat_mols, name='val',
    #         pos_labels=1, ignore_labels=-1, raise_value_error=False))
    #
    #     trainer.extend(E.snapshot_object(
    #         model, "best_val_" + model_filename[task_type]),
    #         trigger=training.triggers.MaxValueTrigger('val/main/roc_auc'))
    # else:
    #     raise NotImplementedError(
    #         'Not implemented task_type = {}'.format(task_type))

    trainer.extend(AutoPrintReport())
    trainer.extend(E.ProgressBar())
    trainer.run()

    # Save the model's parameters.
    model_path = os.path.join(model_dir,  model_filename[task_type])
    print('Saving the trained model to {}...'.format(model_path))
    model.save_pickle(model_path, protocol=args.protocol)


if __name__ == '__main__':
    main()
