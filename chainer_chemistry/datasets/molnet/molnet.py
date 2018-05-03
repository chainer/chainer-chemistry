import gzip
from logging import getLogger
import os
import shutil

import numpy as np
from chainer.dataset import download

from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config #NOQA

_root = 'pfnet/chainer/molnet'

def get_molnet_dataset(dataset_name, preprocessor=None, labels=None,
                       return_smiles=False, target_index=None):
    assert dataset_name in molnet_default_config
    dataset_config = molnet_default_config[dataset_name]
    labels = labels or dataset_config['tasks']
    if isinstance(labels, str):
        labels = [labels, ]

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
        # TODO(motoki): raise Error?
        #
    if dataset_config['dataset_type'] == 'one_file_csv':
        if dataset_config['task_type'] == 'regression':
            postprocess_label = lambda x: np.asarray(x, dtype=np.float32)
        elif dataset_config['task_type'] == 'classification':
            postprocess_label = lambda x: np.asarray(x, dtype=np.int32)

        parser = CSVFileParser(preprocessor, labels=labels,
                               smiles_col=dataset_config['smiles_columns'],
                               postprocess_label=postprocess_label)
        result = parser.parse(get_molnet_filepath(dataset_name),
                              return_smiles=return_smiles,
                              target_index=target_index)
        # TODO(motoki): splitting
        # train, val, test = random_split(result['dataset'])
        # molnet['dataset'] = (train, val, test)
    else:
        raise NotImplementedError
    return result

def get_molnet_filepath(dataset_name, download_if_not_exist=True):
    cache_path = _get_molnet_filepath(dataset_name)
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_molnet_dataset(dataset_name,
                                                    save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path

def _get_molnet_filepath(dataset_name):
    """Construct a filepath which stores QM9 dataset in csv

    This method does not check if the file is already downloaded or not.

    Returns (str): filepath for tox21 dataset

    """
    cache_root = download.get_dataset_directory(_root)
    file_name = molnet_default_config[dataset_name]['url'].split('/')[-1]
    cache_path = os.path.join(cache_root, file_name)
    return cache_path

def download_molnet_dataset(dataset_name, save_filepath):
    logger = getLogger(__name__)
    logger.warning('Download {} dataset, it takes time...'
                   .format(dataset_name))
    dataset_config = molnet_default_config[dataset_name]
    download_file_path = download.cached_download(dataset_config['url'])
    shutil.move(download_file_path, save_filepath)
    # pandas can load gzipped or tarball csv file
    return True
