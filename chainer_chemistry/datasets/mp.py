import os
import pickle
import ast
import shutil
from logging import getLogger

import numpy
import pandas

import chainer
from chainer.dataset import download
import h5py
from pymatgen.core.structure import Structure
from tqdm import tqdm


download_root_url = 'https://chainer-assets.preferred.jp/chainerchem/mp/'
filename_list = ['property_data.csv', 'stability_data.csv', 'cif_data.h5']
_root = 'pfnet/chainer/mp'


def get_mp_filepath(filename, download_if_not_exist=True):
    """Construct a filepath which stores atom_init_json

    This method check whether the file exist or not,  and downloaded it if
    necessary.

    Args:
        filename (str): file name of target file to download.
            See `filename_list` for supported files.
        download_if_not_exist (bool): If `True` download dataset
            if it is not downloaded yet.

    Returns (str): file path for atom_init_json
    """
    if filename not in filename_list:
        raise ValueError('filename {} is not supported!'.format(filename))
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, filename)
    if not os.path.exists(cache_path) and download_if_not_exist:
        logger = getLogger(__name__)
        logger.info('Downloading atom_init.json...')
        download_url = download_root_url + filename
        download_file_path = download.cached_download(download_url)
        shutil.copy(download_file_path, cache_path)
    return cache_path


class MPDataset(chainer.dataset.DatasetMixin):
    """Class for collecting Material Project dataset.

    Refer https://materialsproject.org/ for details.

    Args:
        preprocessor (BasePreprocessor): preprocessor instance
    """

    def __init__(self, preprocessor):
        self.id_prop_data = None
        self.data = None
        self.preprocessor = preprocessor

    def __len__(self):
        """Return the dataset length."""
        return len(self.data)

    def save_pickle(self, path):
        """Save the dataset to pickle object.

        Args:
            path (str): filepath to save dataset.
        """
        print("saving dataset into {}".format(path))
        with open(path, "wb") as file_:
            pickle.dump(self.data, file_)

        return True

    def load_pickle(self, path):
        """Load the dataset from pickle object.

        Args:
            path (str): filepath to load pickle object.
        """
        print("loading dataset from {}".format(path))
        if os.path.exists(path) is False:
            print("Fail.")
            return False
        with open(path, "rb") as file_:
            self.data = pickle.load(file_)

        return True

    def _load_label_list(self, target_list, is_stable=True):
        """Collect the label.

        Args:
            target_list (List): List of target labels.
            is_stable (bool): If this value is true,
                load data that do not have calculation warnings.
        """
        id_prop_data = pandas.read_csv(
            get_mp_filepath("property_data.csv"), index_col=0)
        stability_data = pandas.read_csv(
            get_mp_filepath("stability_data.csv"),
            index_col=0, converters={3: ast.literal_eval})
        id_prop_data = id_prop_data.merge(stability_data, on="material_id")
        # drop data which has warnings
        if is_stable:
            n_warns = numpy.array([len(d) for d in id_prop_data["warnings"]])
            mask = n_warns == 0
            id_prop_data = id_prop_data[mask]

        if "band_gap" in target_list:
            id_prop_data = id_prop_data[id_prop_data["band_gap"].values > 0]

        if "K_VRH" in target_list or "G_VRH" in target_list \
                or "poisson_ratio" in target_list:
            id_prop_data = id_prop_data[id_prop_data["K_VRH"] >= 1]
            id_prop_data = id_prop_data[id_prop_data["G_VRH"] >= 1]

        self.id_prop_data = id_prop_data
        return True

    def get_mp(self, target_list, num_data=None, is_stable=True):
        """Download dataset from Material Project dataset.

        Args:
            target_list (List): List of target labels.
            num_data (int): the number of data that we want to get
            is_stable (bool): If this value is true,
                load data that do not have calculation warnings
        """
        print("loading mp dataset...")
        self._load_label_list(target_list, is_stable)
        cif_data = h5py.File(get_mp_filepath("cif_data.h5"), "r")

        data = self.id_prop_data
        if num_data is not None and num_data >= 0:
            data = data[0:num_data]

        self.data = list()
        data_length = len(data)
        for i in tqdm(range(data_length)):
            # get structure object from CIF
            properties = self.id_prop_data.iloc[i]
            cif_id = properties["material_id"] + ".cif"
            if cif_id not in cif_data:
                continue
            structure = Structure.from_str(cif_data[cif_id].value, "yaml")

            # prepare label
            target = properties[target_list].astype(numpy.float32)
            if numpy.isnan(target).any():
                continue
            # convert unit into /atom
            if "energy" in target:
                n_atom = structure.num_sites
                target["energy"] = target["energy"] / n_atom
            # convert to log10
            if "K_VRH" in target:
                target["K_VRH"] = numpy.log10(target["K_VRH"])
            # convert to log10
            if "G_VRH" in target:
                target["G_VRH"] = numpy.log10(target["G_VRH"])

            self.data.append((structure, target))

        return True

    def get_example(self, i):
        """Return preprocessed dataset from structure object."""
        features = self.preprocessor.get_input_feature_from_crystal(
            self.data[i][0])
        return tuple((*features, self.data[i][1]))
