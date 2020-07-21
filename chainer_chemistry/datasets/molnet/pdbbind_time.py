from logging import getLogger
import os
import shutil

import pandas

from chainer.dataset import download


def get_pdbbind_time():
    """Get time list for PDBbind dataset.

    Args:

    Returns(list):
        Time list for PDBbind dataset.

    """
    df = pandas.read_csv(get_pdbbind_time_filepath(), header=None)
    time_list = df[1].values.tolist()
    return time_list


def get_pdbbind_time_filepath(download_if_not_exist=True):
    """Construct a file path which stores year table of PDBbind.

    This method check whether the file exist or not, and download it if
    necessary.

    Args:
        download_if_not_exist(bool): Download a file if it does not exist.

    Returns(str): filepath for year table

    """
    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' \
          'pdbbind_year.csv'
    file_name = url.split('/')[-1]
    cache_path = _get_pdbbind_time_filepath(file_name)
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_pdbbind_time(url,
                                                  save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_pdbbind_time_filepath(file_name):
    """Construct a filepath which stores year table in csv.

    This method does not check if the file is already downloaded or not.

    Args:
        file_name(str): file name of year table

    Returns(str): filepath for one of year table

    """
    cache_root = download.get_dataset_directory('pfnet/chainer/molnet')
    cache_path = os.path.join(cache_root, file_name)
    return cache_path


def download_pdbbind_time(url, save_filepath):
    """Download and caches PDBBind year table.

    Args:
        url(str): URL of year table
        save_filepath(str): filepath for year table

    Returns(bool): If success downloading, returning `True`.
    """
    download_file_path = download.cached_download(url)
    shutil.move(download_file_path, save_filepath)
    return True
