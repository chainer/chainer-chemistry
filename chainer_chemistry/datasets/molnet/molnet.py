import gzip
from logging import getLogger
import os
import shutil

import numpy as np
from chainer.dataset import download
from chainer.datasets import split_dataset_random

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config #NOQA

_root = 'pfnet/chainer/molnet'

def get_molnet_dataset(dataset_name, preprocessor=None, labels=None,
                       split='random', frac_train=.8, frac_valid=.1,
                       frac_test=.1, seed=777, return_smiles=False,
                       target_index=None):
    assert dataset_name in molnet_default_config
    dataset_config = molnet_default_config[dataset_name]
    labels = labels or dataset_config['tasks']
    if isinstance(labels, str):
        labels = [labels, ]

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
        # TODO(motoki): raise Error?
        #

    if dataset_config['task_type'] == 'regression':
        postprocess_label = lambda x: np.asarray(x, dtype=np.float32)
    elif dataset_config['task_type'] == 'classification':
        def postprocess_label(label_list):
            label_list = np.asarray(label_list)
            label_list[np.isnan(label_list)] = -1
            return label_list.astype(np.int32)

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
            perm = np.random.permutation(len(dataset))
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
    assert filetype in ['onefile', 'train', 'valid', 'test']
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
    """Construct a filepath which stores QM9 dataset in csv

    This method does not check if the file is already downloaded or not.

    Returns (str): filepath for tox21 dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name)
    return cache_path

def download_dataset(dataset_url, save_filepath):
    logger = getLogger(__name__)
    logger.warning('Download {} dataset, it takes time...'
                   .format(dataset_url.split('/')[-1]))
    download_file_path = download.cached_download(dataset_url)
    shutil.move(download_file_path, save_filepath)
    # pandas can load gzipped or tarball csv file
    return True
