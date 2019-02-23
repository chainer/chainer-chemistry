from logging import getLogger
import os
import tarfile
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Union  # NOQA

from chainer.dataset import download
import numpy
import scipy

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

download_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
feat_file_name = 'cora.content'
adj_file_name = 'cora.cites'

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


def get_cora(labels=None, is_normalized=True):
    # type: (Union[str, List[str], None], bool) -> NumpyTupleDataset
    """Download, cache and preprocess Cora dataset.

    Args:
        labels (str or list): List of target labels.
        target_index (list or None): target index list to partially
            extract dataset. If None(default), all examples are parsed.

    Returns:
        dataset, which is composed of `features`

    """
    label_names = labels or get_cora_label_names()
    if isinstance(label_names, str):
        label_names = [
            label_names,
        ]

    features = numpy.genfromtxt(get_cora_filepath()[0], dtype=numpy.dtype(str))
    feat_array = features[:, 1:-1].astype('f')
    if is_normalized:
        feat_array = normalize(feat_array)

    labels = encode_onehot(features[:, -1])
    idx = features[:, 0].astype('i')
    adj = construct_adj_matrix(idx, is_normalized=is_normalized)

    dataset = NumpyTupleDataset(feat_array, adj, labels)
    return dataset


def construct_adj_matrix(
        idx,  # type: numpy.ndarray
        out_size=-1,  # type: int
        self_connection=True,  # type: bool
        is_normalized=True  # type: bool
):
    # type: (...) -> numpy.ndarray
    """Return the adjacent matrix of the given graph.

    This function returns the adjacent matrix of the given graph.

    Args:
        idx (numpy.ndarray):
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of nodes.
            In that case, the adjacent matrix is expanded and zeros are
            padded to right columns and bottom rows.
        self_connection (bool): Add self connection or not.
            If ``True``, diagonal element of adjacency matrix is filled
            with 1.
        is_normalized (bool): Normalize adjacency matrix.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix.
            It is 2-dimensional array with shape (node1, node2), where
            node1 & node2 represent from and to of the edge
            respectively. If ``out_size`` is non-negative, the returned
            its size is equal to that value. Otherwise,
            it is equal to the number of nodes.

    """
    adj_filepath = get_cora_filepath()[1]
    adj = numpy.genfromtxt(adj_filepath).astype('i')
    idx_map = {j: i for i, j in enumerate(idx)}
    adj = numpy.array(list(map(idx_map.get,
                               adj.flatten()))).astype('i').reshape(adj.shape)
    n = len(idx)

    adj_array = numpy.zeros(n**2)
    adj_array[numpy.ravel_multi_index(adj.T, (n, n))] = 1
    adj_array = adj_array.reshape((n, n))

    # build symmetric adjacency matrix
    mask = adj_array.T > adj_array
    adj_array = adj_array + adj_array.T * mask - adj_array * mask

    if self_connection:
        adj_array + numpy.eye(n)

    if is_normalized:
        adj_array = normalize(adj_array)

    if out_size < 0:
        adj_array = adj_array.astype('f')
    elif out_size >= n:
        adj_array = numpy.zeros((out_size, out_size), dtype=numpy.float32)
        adj_array[:n, :n] = adj_array
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of nodes in the input graph (={}).'.format(out_size, n))
    return adj_array


def encode_onehot(labels):
    # type: (numpy.ndarray) -> numpy.ndarray
    classes = get_cora_label_names()
    classes_dict = {
        c: numpy.identity(len(classes))[i, :]
        for i, c in enumerate(classes)
    }
    labels_onehot = numpy.array(
        list(map(classes_dict.get, labels)), dtype=numpy.int32)
    return labels_onehot


def normalize(matrix):
    # type: (numpy.ndarray) -> numpy.ndarray
    rowsum = matrix.sum(axis=1).reshape(-1, 1)
    r_inv = numpy.power(rowsum, -1).flatten()
    r_inv[numpy.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    matrix = r_mat_inv.dot(matrix)
    return matrix


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
        adj_cache_path (str): file path for Cora dataset (adjacency).

    """
    feat_cache_path, adj_cache_path = _get_cora_filepath()
    if not os.path.exists(feat_cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_cora(
                save_dirpath=os.path.dirname(feat_cache_path))
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return feat_cache_path, adj_cache_path


def _get_cora_filepath():
    # type: () -> Tuple[str, str]
    """Construct a filepath which stores Cora dataset.

    This method does not check if the file is already downloaded or not.

    Returns:
        feat_cache_path (str): file path for Cora dataset (features).
        adj_cache_path (str): file path for Cora dataset (adjacency).

    """
    cache_root = download.get_dataset_directory(_root)
    feat_cache_path = os.path.join(cache_root, feat_file_name)
    adj_cache_path = os.path.join(cache_root, adj_file_name)
    return feat_cache_path, adj_cache_path


def download_and_extract_cora(save_dirpath):
    # type: (str) -> bool
    download_file_path = download.cached_download(download_url)
    tf = tarfile.open(download_file_path, 'r')
    tf.extractall(os.path.dirname(save_dirpath))
    return True
