import os

import numpy
import pytest

from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import zinc


ZINC250K_NUM_LABEL = 3
ZINC250K_NUM_DATASET = 249455


def test_get_zinc_filepath_without_download():
    filepath = zinc.get_zinc250k_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = zinc.get_zinc250k_filepath(download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)


@pytest.mark.slow
def test_get_zinc_filepath_with_download():
    filepath = zinc.get_zinc250k_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    # This method downloads the file if not exist
    filepath = zinc.get_zinc250k_filepath(download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)


@pytest.mark.slow
def test_get_zinc():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    dataset = zinc.get_zinc250k(preprocessor=pp)

    # --- Test dataset is correctly obtained ---
    index = numpy.random.choice(len(dataset), None)
    atoms, label = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    assert label.ndim == 1
    assert label.shape[0] == ZINC250K_NUM_LABEL
    assert label.dtype == numpy.float32

    # --- Test number of dataset ---
    assert len(dataset) == ZINC250K_NUM_DATASET


@pytest.mark.slow
def test_get_zinc_smiles():
    # test smiles extraction and dataset order
    pp = AtomicNumberPreprocessor()
    target_index = [0, 7777, 249454]  # set target_index for fast testing...
    dataset, smiles = zinc.get_zinc250k(preprocessor=pp, return_smiles=True,
                                        target_index=target_index)

    # --- Test dataset is correctly obtained ---
    index = numpy.random.choice(len(dataset), None)
    atoms, label = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (atom from, atom to) or (edge_type, atom from, atom to)
    assert label.ndim == 1
    assert label.shape[0] == ZINC250K_NUM_LABEL
    assert label.dtype == numpy.float32

    # --- Test number of dataset ---
    assert len(dataset) == len(target_index)
    assert len(smiles) == len(target_index)

    # --- Test order of dataset ---
    assert smiles[0] == 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    atoms0, labels0 = dataset[0]
    assert numpy.alltrue(atoms0 == numpy.array(
        [6, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 8, 7, 6, 6, 6, 6, 6, 6, 9, 6,
         6], dtype=numpy.int32))
    assert numpy.alltrue(labels0 == numpy.array(
        [5.0506, 0.70201224, 2.0840945], dtype=numpy.float32))

    assert smiles[1] == 'CCCc1cc(NC(=O)Nc2ccc3c(c2)OCCO3)n(C)n1'
    atoms7777, labels7777 = dataset[1]
    assert numpy.alltrue(atoms7777 == numpy.array(
        [6, 6, 6, 6, 6, 6, 7, 6, 8, 7, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 7, 6, 7],
        dtype=numpy.int32))
    assert numpy.alltrue(labels7777 == numpy.array(
        [2.7878, 0.9035222, 2.3195992], dtype=numpy.float32))

    assert smiles[2] == 'O=C(CC(c1ccccc1)c1ccccc1)N1CCN(S(=O)(=O)c2ccccc2[N+](=O)[O-])CC1'  # NOQA
    atoms249454, labels249454 = dataset[2]
    assert numpy.alltrue(atoms249454 == numpy.array(
        [8,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,
         6,  6,  7, 16,  8,  8,  6,  6,  6,  6,  6,  6,  7,  8,  8,  6,  6],
        dtype=numpy.int32))
    assert numpy.alltrue(labels249454 == numpy.array(
        [3.6499, 0.37028658, 2.2142494], dtype=numpy.float32))


def test_get_zinc_label_names():
    label_names = zinc.get_zinc250k_label_names()
    assert isinstance(label_names, list)
    for label in label_names:
        assert isinstance(label, str)


if __name__ == '__main__':
    args = [__file__, '-v', '-s']
    pytest.main(args=args)
