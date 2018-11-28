from logging import getLogger
import os

from chainer.dataset import download
import numpy
import pandas

from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA

download_url = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'  # NOQA
file_name_250k = 'zinc250k.csv'

_root = 'pfnet/chainer/zinc'

_label_names = ['logP', 'qed', 'SAS']
_smiles_column_names = ['smiles']


def get_zinc250k_label_names():
    """Returns label names of ZINC250k datasets."""
    return _label_names


def get_zinc250k(preprocessor=None, labels=None, return_smiles=False,
                 target_index=None):
    """Downloads, caches and preprocesses Zinc 250K dataset.

    Args:
        preprocessor (BasePreprocessor): Preprocessor.
            This should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.
        target_index (list or None): target index list to partially extract
            dataset. If None (default), all examples are parsed.

    Returns:
        dataset, which is composed of `features`, which depends on
        `preprocess_method`.

    """
    labels = labels or get_zinc250k_label_names()
    if isinstance(labels, str):
        labels = [labels, ]

    def postprocess_label(label_list):
        # This is regression task, cast to float value.
        return numpy.asarray(label_list, dtype=numpy.float32)

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='smiles')
    result = parser.parse(get_zinc250k_filepath(), return_smiles=return_smiles,
                          target_index=target_index)

    if return_smiles:
        return result['dataset'], result['smiles']
    else:
        return result['dataset']


def get_zinc250k_filepath(download_if_not_exist=True):
    """Construct a filepath which stores ZINC250k dataset for config_name

    This method check whether the file exist or not,  and downloaded it if
    necessary.

    Args:
        download_if_not_exist (bool): If `True` download dataset
            if it is not downloaded yet.

    Returns (str): file path for ZINC250k dataset (csv format)

    """
    cache_path = _get_zinc250k_filepath()
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_zinc250k(
                save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_zinc250k_filepath():
    """Construct a filepath which stores ZINC250k dataset in csv

    This method does not check if the file is already downloaded or not.

    Returns (str): filepath for ZINC250k dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name_250k)
    return cache_path


def _remove_new_line(s):
    return s.replace('\n', '')


def download_and_extract_zinc250k(save_filepath):
    logger = getLogger(__name__)
    logger.info('Extracting ZINC250k dataset...')
    download_file_path = download.cached_download(download_url)
    df = pandas.read_csv(download_file_path)
    # 'smiles' column contains '\n', need to remove it.
    df['smiles'] = df['smiles'].apply(_remove_new_line)
    df.to_csv(save_filepath, columns=_smiles_column_names + _label_names)
    return True
