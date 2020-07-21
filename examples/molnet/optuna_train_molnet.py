#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy
import os
import types

import pickle

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training

from chainer.training import extensions as E

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.preprocessors import preprocess_method_dict, neighbor_label_expansion
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.links import StandardScaler
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.models.prediction import Regressor
from chainer_chemistry.models.prediction import set_up_predictor
from chainer_chemistry.training.extensions import BatchEvaluator, ROCAUCEvaluator  # NOQA
from chainer_chemistry.training.extensions.auto_print_report import AutoPrintReport  # NOQA
from chainer_chemistry.models.cnle.cnle_graph_conv_model import MAX_NLE_NUM

import optuna


def parse_arguments():
    # Lists of supported preprocessing methods/models and datasets.
    method_list = ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn', 'relgcn',
                   'relgat', 'gin', 'gnnfilm',
                   'nfp_gwm', 'ggnn_gwm', 'rsgcn_gwm', 'gin_gwm',
                   'nfp_cnle', 'ggnn_cnle',  'relgat_cnle', 'relgcn_cnle', 'rsgcn_cnle', 'gin_cnle',
                   'nfp_gcnle', 'ggnn_gcnle',  'relgat_gcnle', 'relgcn_gcnle', 'rsgcn_gcnle', 'gin_gcnle']
    dataset_names = list(molnet_default_config.keys())
    scale_list = ['standardize', 'none']

    parser = argparse.ArgumentParser(description='optuna optimization for molnet example: we optimize unit-num (16-512), conv-layers (2-9), and adam_alpha (1e-5-1e-1)')
    parser.add_argument('--method', '-m', type=str, choices=method_list,
                        help='method name', default='nfp')
    parser.add_argument('--label', '-l', type=str, default='',
                        help='target label for regression; empty string means '
                        'predicting all properties at once')
    #parser.add_argument('--conv-layers', '-c', type=int, default=4, help='number of convolution layers')
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
    #parser.add_argument('--unit-num', '-u', type=int, default=16, help='number of units in one layer of the model')
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
    parser.add_argument('--adam-alpha', type=float, help='alpha of adam', default=0.001)

    parser.add_argument('--apply-nle', action='store_true', help="Enable to apply Neighbor Label Expansion")
    parser.add_argument('--cutoff-nle', type=str, default=0, help="set more than zero to cut-off Neighbor Label Expansion")
    parser.add_argument('--apply-cwle', action='store_true', help="Enable to apply Combined Neighbor Label Expansion")
    parser.add_argument('--apply-gwle', action='store_true', help="Enable to apply Gated Combined Neighbor Label Expansion")
    parser.add_argument('--num-hop', '-k', type=int, default=1, help="The number of iterations for NLE and CNLE")
    parser.add_argument('--num-trial', type=int, default=100, help="The number of trials of optuna")

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


def download_entire_dataset(dataset_name, num_data, labels, method, cache_dir, apply_nle_flag=False, cutoff_nle=0, apply_cnle_flag=False, apply_gcnle_flag=False, n_hop=1):
    """Downloads the train/valid/test parts of a dataset and stores them in the
    cache directory.
    Args:
        dataset_name: Dataset to be downloaded.
        num_data: Amount of data samples to be parsed from the dataset.
        labels: Target labels for regression.
        method: Method name. See `parse_arguments`.
        cache_dir: Directory to store the dataset to.
        apply_nle_flag: boolean, set True if you apply neighbor label expansion (NLE)
        cutoff_nle: int set more than zero to cut off NLEs
        apply_cnle_flag: boolean, set True if you apply Combined neighbor label expansion (CNLE)
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

    # apply Neighboring Label Expansion
    if apply_nle_flag:
        # print("type(dataset_parts)="  + str(dataset_parts)) # should be list
        dataset_parts_expand, labels_expanded, labels_frequency = neighbor_label_expansion.apply_nle_for_datasets(dataset_parts, cutoff_nle, n_hop)
        dataset_parts = dataset_parts_expand
        num_expanded_symbols = len(labels_expanded)
        print("Neighbor Node Expansion Applied to datasets: vocab=", num_expanded_symbols)
        print(labels_expanded)

        # save in text
        file_name = "NLE_labels.dat"
        path = os.path.join(cache_dir, file_name)
        with open(path, "w") as fout:
            for label in labels_expanded:
                fout.write(label + " " + str(labels_frequency[label]) + "\n")

        # save binaries
        file_name = "NLE_labels.pkl"
        outfile = cache_dir + "/" + file_name
        with open(outfile, "wb") as fout:
            pickle.dump( (labels_expanded, labels_frequency), fout)

    elif apply_cnle_flag:
        # ToDo; extend eac hdataset with (atom array, nle array, adjacency tensor).

        # print("type(dataset_parts)="  + str(dataset_parts)) # should be list
        dataset_parts_expand, labels_expanded, labels_frequency = neighbor_label_expansion.apply_cnle_for_datasets(dataset_parts, n_hop)
        dataset_parts = dataset_parts_expand
        num_expanded_symbols = len(labels_expanded)
        print("Combined Neighbor Node Expansion Applied to datasets: vocab=", num_expanded_symbols)
        print(labels_expanded)

        # save in text
        file_name = "CNLE_labels.dat"
        path = os.path.join(cache_dir, file_name)
        with open(path, "w") as fout:
            for label in labels_expanded:
                fout.write(label + " " + str(labels_frequency[label]) + "\n")

        # save binaries
        file_name = "CNLE_labels.pkl"
        outfile = cache_dir + "/" + file_name
        with open(outfile, "wb") as fout:
            pickle.dump( (labels_expanded, labels_frequency), fout)

    elif apply_gcnle_flag:
        # ToDo; extend eac hdataset with (atom array, nle array, adjacency tensor).

        # print("type(dataset_parts)="  + str(dataset_parts)) # should be list
        dataset_parts_expand, labels_expanded, labels_frequency = neighbor_label_expansion.apply_cnle_for_datasets(dataset_parts, n_hop)
        dataset_parts = dataset_parts_expand
        num_expanded_symbols = len(labels_expanded)
        print("Combined Neighbor Node Expansion Applied to datasets: vocab=", num_expanded_symbols)
        print(labels_expanded)

        # save in text
        file_name = "CNLE_labels.dat"
        path = os.path.join(cache_dir, file_name)
        with open(path, "w") as fout:
            for label in labels_expanded:
                fout.write(label + " " + str(labels_frequency[label]) + "\n")

        # save binaries
        file_name = "CNLE_labels.pkl"
        outfile = cache_dir + "/" + file_name
        with open(outfile, "wb") as fout:
            pickle.dump( (labels_expanded, labels_frequency), fout)


    else:
        labels_expanded = []

    # ToDO: scaler should be placed here
    # ToDo: fit the scaler
    # ToDo: transform dataset_parts[0-2]

    for i, part in enumerate(['train', 'valid', 'test']):
        filename = dataset_part_filename(part, num_data)
        path = os.path.join(cache_dir, filename)
        if False:
            print(type(dataset_parts[i]))
            print(type(dataset_parts[i][0]))
            print(type(dataset_parts[i][0][0]))
            print(type(dataset_parts[i][0][1]))
            print(type(dataset_parts[i][0][2]))
            print(dataset_parts[i][0][0].shape)
            print(dataset_parts[i][0][1].shape)
            print(dataset_parts[i][0][2].shape)
            print(dataset_parts[i][0][0].dtype)
            print(dataset_parts[i][0][1].dtype)
            print(dataset_parts[i][0][2].dtype)
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
    print(args)

    # call objectives and get the valiation score
    #study = optuna.create_study(direction='minimize')
    #study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    obj = lambda x: objective(x, args)
    if int(args.device) >= 0:
        import cupy
        catch=(ValueError, cupy.cuda.runtime.CUDARuntimeError)
    else:
        catch=(ValueError,)
    study.optimize(obj, n_trials=args.num_trial, catch=catch)

    print('Number of finished trials: ', len(study.trials))

    from pandas import DataFrame
    df = study.trials_dataframe()
    # dump all the trials
    df.to_csv(args.out + "_" + args.method + "_" + args.dataset + "_BO_alltrials.tsv", sep="\t")

    print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))
    # dump the best BO trials in a text file amd the STDOUT
    with open(args.out + "_" + args.method + "_" + args.dataset + "_BO_hyprm.txt", 'w') as fout:
        print('Number of finished trials: ', len(study.trials))
        fout.write("Number of finished trials: " + str(len(study.trials)) + "\n")

        print('Best trial:')
        fout.write('Best trial:\n')
        trial = study.best_trial

        print('  Value: ', trial.value)
        fout.write('  Value: ' +  str(trial.value) + "\n")

        print('  Params: ')
        fout.write('  Params: \n')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
            fout.write('    {}: {}\n'.format(key, value))
        # end-for

        print('  User attrs:')
        fout.write('  User attrs:\n')
        for key, value in trial.user_attrs.items():
            print('    {}: {}'.format(key, value))
            fout.write('    {}: {}\n'.format(key, value))
        # end-for
    # end-with

# end main()


def objective(trial, args):

    # Set up some useful variables that will be used later on.
    dataset_name = args.dataset
    method = args.method
    num_data = args.num_data
    # n_unit = args.unit_num
    # conv_layers = args.conv_layers
    # adam_alpha = args.adam_alpha
    apply_nle_flag = args.apply_nle
    cutoff_nle = args.cutoff_nle
    apply_cnle_flag = args.apply_cnle
    apply_gcnle_flag = args.apply_gcnle

    #n_unit = int(trial.suggest_loguniform('n_unit', 16, 256))
    #conv_layers = int(trial.suggest_uniform('conv_layers', 2, 6))
    # For RelGAT
    n_unit = int(trial.suggest_loguniform('n_unit', 4, 24))
    conv_layers = int(trial.suggest_uniform('conv_layers', 2, 4))
    adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)


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
                 for p in ['train', 'valid', 'test']]


    # ToDo: We need to incoporeat scaler into download_entire_dataset, instead of predictors. 
    paths = [os.path.join(cache_dir, f) for f in filenames]
    if all([os.path.exists(path) for path in paths]):
        dataset_parts = []
        for path in paths:
            print('Loading cached dataset from {}.'.format(path))
            dataset_parts.append(NumpyTupleDataset.load(path))
    else:
        dataset_parts = download_entire_dataset(dataset_name, num_data, labels,
                                                method, cache_dir, apply_nle_flag, cutoff_nle, apply_cnle_flag, apply_gcnle_flag, args.num_hop)
    train, valid = dataset_parts[0], dataset_parts[1]

    # ToDo: scaler must be incorporated into download_entire_datasets. not here
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

    # ToDo: set label_scaler always None
    # Set up the predictor.
    if apply_nle_flag:
        # find the num_atoms
        max_symbol_index = neighbor_label_expansion.findmaxidx(dataset_parts)
        print("number of expanded symbols=", max_symbol_index)
        predictor = set_up_predictor(
            method, n_unit, conv_layers, class_num,
            label_scaler=scaler, n_atom_types=max_symbol_index)
    elif apply_cnle_flag or apply_gcnle_flag:
        n_nle_types = neighbor_label_expansion.findmaxidx(
            dataset_parts, 'nle_label')
        # Kenta Oono (oono@preferre.jp)
        # In the previous implementation, we use MAX_NLE_NUM
        # as the dimension of one-hot vectors for NLE labels
        # when the model is CNLE or WLNE and hop_num k = 1.
        # When k >= 2, # of nle labels can be larger than MAX_NLE_NUM,
        # which causes an error.
        # Therefore, we have increased the dimension of vectors.
        # To align with the previous experiments,
        # we change n_nle_types only if it exceeds MAX_NLE_NUM.
        n_nle_types = max(n_nle_types, MAX_NLE_NUM)
        print("number of expanded symbols (CNLE/WCNLE) = ", n_nle_types)
        predictor = set_up_predictor(
            method, n_unit, conv_layers, class_num,
            label_scaler=scaler, n_nle_types=n_nle_types)
    else:
        predictor = set_up_predictor(
            method, n_unit, conv_layers, class_num,
            label_scaler=scaler)


    # Set up the iterators.
    train_iter = iterators.SerialIterator(train, args.batchsize)
    valid_iter = iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)

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
    optimizer = optimizers.Adam(alpha=adam_alpha)
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

    log_report_extension = E.LogReport()
    trainer.extend(log_report_extension)
    print_report_targets = ['epoch', 'main/loss', 'validation/main/loss']
    for metric_name, metric_fun in metrics.items():
        if isinstance(metric_fun, types.FunctionType):
            print_report_targets.append('main/' + metric_name)
            print_report_targets.append('validation/main/' + metric_name)
        elif issubclass(metric_fun, BatchEvaluator):
            trainer.extend(metric_fun(
                valid_iter, model, device=args.device, eval_func=predictor,
                converter=concat_mols, name='val',
                raise_value_error=False))
            print_report_targets.append('val/main/' + metric_name)
        else:
            raise TypeError('{} is not supported for metrics function.'
                            .format(type(metrics_fun)))
    print_report_targets.append('elapsed_time')

    if task_type == 'regression':
        pass
        #trainer.extend(optuna.integration.ChainerPruningExtension(trial, 'validation/main/MAE', (7, 'epoch')))
    elif task_type == 'classification':
        train_eval_iter = iterators.SerialIterator(train, args.batchsize,
                                                   repeat=False, shuffle=False)
        trainer.extend(ROCAUCEvaluator(
            train_eval_iter, predictor, eval_func=predictor,
            device=args.device, converter=concat_mols, name='train',
            pos_labels=1, ignore_labels=-1, raise_value_error=False))
        trainer.extend(ROCAUCEvaluator(
            valid_iter, predictor, eval_func=predictor,
            device=args.device, converter=concat_mols, name='val',
            pos_labels=1, ignore_labels=-1, raise_value_error=False))
        print_report_targets.append('main/accuracy')
        print_report_targets.append('train/main/roc_auc')
        print_report_targets.append('validation/main/loss')
        print_report_targets.append('val/main/roc_auc')
    else:
        raise NotImplementedError(
            'Not implemented task_type = {}'.format(task_type))

    trainer.extend(optuna.integration.ChainerPruningExtension(trial, 'validation/main/loss', (5, 'epoch')))
    trainer.extend(E.PrintReport(print_report_targets))


    # pruner


    #trainer.extend(E.snapshot(), trigger=(args.epoch, 'epoch'))
    #trainer.extend(E.ProgressBar())
    trainer.run()

    #
    # return the target value.
    #

    # return the objective function for optuna.optimize
    all_logs = log_report_extension.log
    if task_type == 'regression':
        MAEs = []
        for log in all_logs:
            MAEs.append(log['validation/main/MAE'])
        MAE = numpy.mean(MAEs)

        if MAE < 0.000001:
            MAE = 100000000 # reject
        return MAE
    elif task_type == 'classification':
        AUCs = []
        for log in all_logs:
            AUCs.append(log['val/main/roc_auc'])
        AUC = numpy.max(AUCs)

        return -1.0 * AUC



# end of objective()

if __name__ == '__main__':
    main()
