from logging import getLogger
import os
import tarfile
from typing import List, Tuple  # NOQA

from chainer.dataset import download

download_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
feat_file_name = 'cora.content'
edge_file_name = 'cora.cites'

_root = 'pfnet/chainer/cora'

_label_names = [
    'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
    'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
    'Theory'
]


def get_cora_label_names():
    # type: () -> List[str]
    """Return label names of Cora dataset."""
    return _label_names


def get_cora_dirpath(download_if_not_exist=True):
    # type: (bool) -> str
    """Construct a dirpath which stores Cora dataset.

    This method check whether the file exist or not, and downloaded it
    if necessary.

    Args:
        download_if_not_exist (bool): If ``True``, download dataset
            if it is not downloaded yet.

    Returns:
        dirpath (str): directory path for Cora dataset.
    """
    feat_cache_path, edge_cache_path = get_cora_filepath(
        download_if_not_exist=download_if_not_exist)
    dirpath = os.path.dirname(feat_cache_path)
    dirpath2 = os.path.dirname(edge_cache_path)
    assert dirpath == dirpath2
    return dirpath


def get_cora_filepath(download_if_not_exist=True):
    # type: (bool) -> Tuple[str, str]
    """Construct a filepath which stores Cora dataset.

    This method check whether the file exist or not, and downloaded it
    if necessary.

    Args:
        download_if_not_exist (bool): If ``True``, download dataset
            if it is not downloaded yet.

    Returns:
        feat_cache_path (str): file path for Cora dataset (features).
        edge_cache_path (str): file path for Cora dataset (edge index).
    """
    feat_cache_path, edge_cache_path = _get_cora_filepath()
    if not os.path.exists(feat_cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_cora(
                save_dirpath=os.path.dirname(feat_cache_path))
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return feat_cache_path, edge_cache_path


def _get_cora_filepath():
    # type: () -> Tuple[str, str]
    """Construct a filepath which stores Cora dataset.

    This method does not check if the file is already downloaded or not.

    Returns:
        feat_cache_path (str): file path for Cora dataset (features).
        edge_cache_path (str): file path for Cora dataset (edge index).
    """
    cache_root = download.get_dataset_directory(_root)
    feat_cache_path = os.path.join(cache_root, feat_file_name)
    edge_cache_path = os.path.join(cache_root, edge_file_name)
    return feat_cache_path, edge_cache_path


def download_and_extract_cora(save_dirpath):
    # type: (str) -> bool
    print('downloading cora dataset...')
    download_file_path = download.cached_download(download_url)
    print('extracting cora dataset...')
    tf = tarfile.open(download_file_path, 'r')
    tf.extractall(os.path.dirname(save_dirpath))
    return True
