from logging import getLogger
import os
from zipfile import ZipFile

import networkx as nx
import numpy
import scipy

from chainer.dataset import download

download_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip'
feat_file_name = 'reddit_data.npz'
edge_file_name = 'reddit_graph.npz'

_root = 'pfnet/chainer/reddit'


def reddit_to_networkx(dirpath):
    print("Loading graph data")
    coo_adj = scipy.sparse.load_npz(os.path.join(dirpath, edge_file_name))
    G = nx.from_scipy_sparse_matrix(coo_adj)

    print("Loading node feature and label")
    # node feature, edge label
    reddit_data = numpy.load(os.path.join(dirpath, feat_file_name))
    G.graph['x'] = reddit_data['feature'].astype(numpy.float32)
    G.graph['y'] = reddit_data['label'].astype(numpy.int32)

    G.graph['label_num'] = 41
    # G = nx.convert_node_labels_to_integers(G)
    print("Finish loading graph: {}".format(dirpath))
    return G


def get_reddit_dirpath(download_if_not_exist=True):
    # type: (bool) -> str
    """Construct a dirpath which stores reddit dataset.

    This method check whether the file exist or not, and downloaded it
    if necessary.

    Args:
        download_if_not_exist (bool): If ``True``, download dataset
            if it is not downloaded yet.

    Returns:
        dirpath (str): directory path for reddit dataset.
    """
    feat_cache_path, edge_cache_path = get_reddit_filepath(
        download_if_not_exist=download_if_not_exist)
    dirpath = os.path.dirname(feat_cache_path)
    dirpath2 = os.path.dirname(edge_cache_path)
    assert dirpath == dirpath2
    return dirpath


def get_reddit_filepath(download_if_not_exist=True):
    # type: (bool) -> Tuple[str, str]
    """Construct a filepath which stores reddit dataset.

    This method check whether the file exist or not, and downloaded it
    if necessary.

    Args:
        download_if_not_exist (bool): If ``True``, download dataset
            if it is not downloaded yet.
    Returns:
        feat_cache_path (str): file path for reddit dataset (features).
        edge_cache_path (str): file path for reddit dataset (edge index).
    """
    feat_cache_path, edge_cache_path = _get_reddit_filepath()
    if not os.path.exists(feat_cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_reddit(
                save_dirpath=os.path.dirname(feat_cache_path))
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return feat_cache_path, edge_cache_path


def _get_reddit_filepath():
    # type: () -> Tuple[str, str]
    """Construct a filepath which stores reddit dataset.

    This method does not check if the file is already downloaded or not.

    Returns:
        feat_cache_path (str): file path for reddit dataset (features).
        edge_cache_path (str): file path for reddit dataset (edge index).
    """
    cache_root = download.get_dataset_directory(_root)
    feat_cache_path = os.path.join(cache_root, feat_file_name)
    edge_cache_path = os.path.join(cache_root, edge_file_name)
    return feat_cache_path, edge_cache_path


def download_and_extract_reddit(save_dirpath):
    # type: (str) -> bool
    print('downloading reddit dataset...')
    download_file_path = download.cached_download(download_url)
    print('extracting reddit dataset...')
    zip = ZipFile(download_file_path, 'r')
    zip.extractall(save_dirpath)
    return True
