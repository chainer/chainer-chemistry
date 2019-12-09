from logging import getLogger
import numpy
import os
import shutil

from chainer.dataset import download

from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from chainer_chemistry.dataset.utils import GaussianDistance
from chainer_chemistry.utils import load_json

download_url = 'https://raw.githubusercontent.com/txie-93/cgcnn/master/data/sample-regression/atom_init.json'  # NOQA
file_name_atom_init_json = 'atom_init.json'

_root = 'pfnet/chainer/cgcnn'


def get_atom_init_json_filepath(download_if_not_exist=True):
    """Construct a filepath which stores atom_init_json

    This method check whether the file exist or not,  and downloaded it if
    necessary.

    Args:
        download_if_not_exist (bool): If `True` download dataset
            if it is not downloaded yet.

    Returns (str): file path for atom_init_json
    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name_atom_init_json)
    if not os.path.exists(cache_path) and download_if_not_exist:
        logger = getLogger(__name__)
        logger.info('Downloading atom_init.json...')
        download_file_path = download.cached_download(download_url)
        shutil.copy(download_file_path, cache_path)
    return cache_path


class CGCNNPreprocessor(MolPreprocessor):
    """CGCNNPreprocessor

    Args:
    For Molecule: TODO
    """

    def __init__(self, max_num_nbr=12, max_radius=8, expand_dim=40):
        super(CGCNNPreprocessor, self).__init__()

        self.max_num_nbr = max_num_nbr
        self.max_radius = max_radius
        self.gdf = GaussianDistance(centers=numpy.linspace(0, 8, expand_dim))
        feat_dict = load_json(get_atom_init_json_filepath())
        self.atom_features = {int(key): numpy.array(value,
                                                    dtype=numpy.float32)
                              for key, value in feat_dict.items()}

    def get_input_features(self, mol):
        raise NotImplementedError()
