import os

import numpy
import pytest

from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import qm9


QM9_NUM_LABEL = 15
QM9_NUM_DATASET = 133885


def test_get_qm9_filepath_without_download():
    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)


@pytest.mark.slow
def test_get_qm9_filepath_with_download():
    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    # This method downloads the file if not exist
    filepath = qm9.get_qm9_filepath(download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)


@pytest.mark.slow
def test_get_qm9():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    dataset = qm9.get_qm9(preprocessor=pp)

    # --- Test dataset is correctly obtained ---
    index = numpy.random.choice(len(dataset), None)
    atoms, label = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (atom from, atom to) or (edge_type, atom from, atom to)
    assert label.ndim == 1
    assert label.shape[0] == QM9_NUM_LABEL
    assert label.dtype == numpy.float32

    # --- Test number of dataset ---
    assert len(dataset) == QM9_NUM_DATASET


@pytest.mark.slow
def test_get_qm9_smiles():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    dataset, smiles = qm9.get_qm9(preprocessor=pp, return_smiles=True)

    # --- Test dataset is correctly obtained ---
    index = numpy.random.choice(len(dataset), None)
    atoms, label = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (atom from, atom to) or (edge_type, atom from, atom to)
    assert label.ndim == 1
    assert label.shape[0] == QM9_NUM_LABEL
    assert label.dtype == numpy.float32

    # --- Test number of dataset ---
    assert len(dataset) == QM9_NUM_DATASET
    assert len(smiles) == QM9_NUM_DATASET

    # --- Test order of dataset ---
    atoms0, labels0 = dataset[0]
    assert smiles[0] == 'C'
    assert numpy.alltrue(atoms0 == numpy.array([6], dtype=numpy.int32))

    atoms7777, labels7777 = dataset[7777]
    assert smiles[7777] == 'CC1=NCCC(C)O1'
    assert numpy.alltrue(
        atoms7777 == numpy.array([6, 6, 7, 6, 6, 6, 6, 8], dtype=numpy.int32))

    atoms133884, labels133884 = dataset[133884]
    assert smiles[133884] == 'C1N2C3C4C5OC13C2C54'
    assert numpy.alltrue(
        atoms133884 == numpy.array([6, 7, 6, 6, 6, 8, 6, 6, 6],
                                   dtype=numpy.int32))


def test_get_qm9_label_names():
    label_names = qm9.get_qm9_label_names()
    assert isinstance(label_names, list)
    for label in label_names:
        assert isinstance(label, str)


if __name__ == '__main__':
    args = [__file__, '-v', '-s']
    pytest.main(args=args)
