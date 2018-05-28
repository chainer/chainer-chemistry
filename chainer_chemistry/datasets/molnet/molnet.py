from logging import getLogger
import os
import shutil

import numpy
from chainer.dataset import download

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config # NOQA

_root = 'pfnet/chainer/molnet'


def get_molnet_dataset(dataset_name, preprocessor=None, labels=None,
                       split='random', frac_train=.8, frac_valid=.1,
                       frac_test=.1, seed=777, return_smiles=False,
                       target_index=None):
    """Downloads, caches and preprocess MoleculeNet dataset.

    Args:
        dataset_name (str): MoleculeNet dataset name. If you want to know the
            detail of MoleculeNet, please refer to
            `official site <http://moleculenet.ai/datasets-1>`_
            If you would like to know what dataset_name is available for
            chainer_chemistry, please refer to `molnet_config.py`.
        preprocessor (BasePreprocessor): Preprocessor.
            It should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.
        target_index (list or None): target index list to partially extract
            dataset. If `None` (default), all examples are parsed.
    Returns (dict):
        Dictionary that contains dataset that is already splitted into train,
        valid and test dataset and 1-d numpy array with dtype=object(string)
        which is a vector of smiles for each example or `None`.

    """
    if dataset_name not in molnet_default_config:
        raise ValueError("We don't support {} dataset. Please choose from {}".
                         format(dataset_name,
                                list(molnet_default_config.keys())))
    dataset_config = molnet_default_config[dataset_name]
    labels = labels or dataset_config['tasks']
    if isinstance(labels, str):
        labels = [labels, ]

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()

    if dataset_config['task_type'] == 'regression':
        def postprocess_label(label_list):
            return numpy.asarray(label_list, dtype=numpy.float32)
    elif dataset_config['task_type'] == 'classification':
        def postprocess_label(label_list):
            label_list = numpy.asarray(label_list)
            label_list[numpy.isnan(label_list)] = -1
            return label_list.astype(numpy.int32)

    parser = CSVFileParser(preprocessor, labels=labels,
                           smiles_col=dataset_config['smiles_columns'],
                           postprocess_label=postprocess_label)
    if dataset_config['dataset_type'] == 'one_file_csv':
        result = parser.parse(get_molnet_filepath(dataset_name),
                              return_smiles=return_smiles,
                              target_index=target_index)
        # TODO(motoki): splitting function or class
        dataset = result['dataset']
        if split == 'random':
            perm = numpy.random.permutation(len(dataset))
            dataset = NumpyTupleDataset(*dataset.features[perm])
            train_data_size = int(len(dataset) * frac_train)
            valid_data_size = int(len(dataset) * frac_valid)
            train = NumpyTupleDataset(*dataset.features[:train_data_size])
            valid = NumpyTupleDataset(*dataset.features[train_data_size:
                                                        train_data_size +
                                                        valid_data_size])
            test = NumpyTupleDataset(*dataset.features[train_data_size +
                                                       valid_data_size:])

            result['dataset'] = (train, valid, test)
            if return_smiles:
                smiles = result['smiles'][perm]
                train_smiles = smiles[:train_data_size]
                valid_smiles = smiles[train_data_size:train_data_size +
                                      valid_data_size]
                test_smiles = smiles[train_data_size + valid_data_size:]
                result['smiles'] = (train_smiles, valid_smiles, test_smiles)
            else:
                result['smiles'] = None
        else:
            raise NotImplementedError
    elif dataset_config['dataset_type'] == 'separate_csv':
        result = {}
        train_result = parser.parse(get_molnet_filepath(dataset_name, 'train'),
                                    return_smiles=return_smiles,
                                    target_index=target_index)
        valid_result = parser.parse(get_molnet_filepath(dataset_name, 'valid'),
                                    return_smiles=return_smiles,
                                    target_index=target_index)
        test_result = parser.parse(get_molnet_filepath(dataset_name, 'test'),
                                   return_smiles=return_smiles,
                                   target_index=target_index)
        result['dataset'] = (train_result['dataset'], valid_result['dataset'],
                             test_result['dataset'])
        result['smiles'] = (train_result['smiles'], valid_result['smiles'],
                            test_result['smiles'])
    else:
        raise NotImplementedError
    return result


def get_molnet_filepath(dataset_name, filetype='onefile',
                        download_if_not_exist=True):
    """Construct a file path which stores MoleculeNet dataset.

    This method check whether the file exist or not, and downloaded it if
    necessary.

    Args:
        dataset_name (str): MoleculeNet dataset name.
        file_type (str): either 'onefile', 'train', 'valid', 'test'
        download_if_not_exist (bool): Download a file if it does not exist.

    Returns (str): filepath for specific MoleculeNet dataset

    """
    if filetype not in ['onefile', 'train', 'valid', 'test']:
        raise ValueError("Please choose filetype from {}".format(
            ['onefile', 'train', 'valid', 'test']))
    if filetype == 'onefile':
        url_key = 'url'
    else:
        url_key = filetype + '_url'
    file_url = molnet_default_config[dataset_name][url_key]
    file_name = file_url.split('/')[-1]
    cache_path = _get_molnet_filepath(file_name)
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_dataset(file_url,
                                             save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_molnet_filepath(file_name):
    """Construct a filepath which stores MoleculeNet dataset in csv

    This method does not check if the file is already downloaded or not.

    Args:
        file_name (str): file name of MoleculeNet dataset

    Returns (str): filepath for one of MoleculeNet dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name)
    return cache_path


def download_dataset(dataset_url, save_filepath):
    """Download and caches MoleculeNet Dataset

    Args:
        dataset_url (str): URL of dataset
        save_filepath (str): filepath for dataset

    Returns (bool): If success downloading, returning `True`.

    """
    logger = getLogger(__name__)
    logger.warning('Downloading {} dataset, it takes time...'
                   .format(dataset_url.split('/')[-1]))
    download_file_path = download.cached_download(dataset_url)
    shutil.move(download_file_path, save_filepath)
    # pandas can load gzipped or tarball csv file
    return True
