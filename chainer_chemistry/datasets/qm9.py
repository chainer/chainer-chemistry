import glob
from logging import getLogger
import os
import shutil
import tarfile
import tempfile

from chainer.dataset import download
import numpy
import pandas
from tqdm import tqdm

from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA

download_url = 'https://ndownloader.figshare.com/files/3195389'
file_name = 'qm9.csv'

_root = 'pfnet/chainer/qm9'

_label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                'zpve', 'U0', 'U', 'H', 'G', 'Cv']
_smiles_column_names = ['SMILES1', 'SMILES2']


def get_qm9_label_names():
    """Returns label names of QM9 datasets."""
    return _label_names


def get_qm9(preprocessor=None, labels=None, return_smiles=False):
    """Downloads, caches and preprocesses QM9 dataset.

    Args:
        preprocessor (BasePreprocessor): Preprocessor.
            This should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.

    Returns:
        dataset, which is composed of `features`, which depends on
        `preprocess_method`.

    """
    labels = labels or get_qm9_label_names()
    if isinstance(labels, str):
        labels = [labels, ]

    def postprocess_label(label_list):
        # This is regression task, cast to float value.
        return numpy.asarray(label_list, dtype=numpy.float32)

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES1')
    result = parser.parse(get_qm9_filepath(), return_smiles=return_smiles)

    if return_smiles:
        return result['dataset'], result['smiles']
    else:
        return result['dataset']


def get_qm9_filepath(download_if_not_exist=True):
    """Construct a filepath which stores qm9 dataset for config_name

    This method check whether the file exist or not,  and downloaded it if
    necessary.

    Args:
        config_name: either 'train', 'val', or 'test'

    Returns (str): filepath for qm9 dataset

    """
    cache_path = _get_qm9_filepath()
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_qm9(save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_qm9_filepath():
    """Construct a filepath which stores QM9 dataset in csv

    This method does not check if the file is already downloaded or not.

    Returns (str): filepath for tox21 dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name)
    return cache_path


def download_and_extract_qm9(save_filepath):
    logger = getLogger(__name__)
    logger.warning('Extracting QM9 dataset, it takes time...')
    download_file_path = download.cached_download(download_url)
    tf = tarfile.open(download_file_path, 'r')
    temp_dir = tempfile.mkdtemp()
    tf.extractall(temp_dir)
    file_re = os.path.join(temp_dir, '*.xyz')
    file_pathes = glob.glob(file_re)
    # Make sure the order is sorted
    file_pathes.sort()
    ls = []
    for path in tqdm(file_pathes):
        with open(path, 'r') as f:
            data = [line.strip() for line in f]

        num_atom = int(data[0])
        properties = list(map(float, data[1].split('\t')[1:]))
        smiles = data[3 + num_atom].split('\t')
        new_ls = smiles + properties
        ls.append(new_ls)

    df = pandas.DataFrame(ls, columns=_smiles_column_names + _label_names)
    df.to_csv(save_filepath)
    shutil.rmtree(temp_dir)
    return True
