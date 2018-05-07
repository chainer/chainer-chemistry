import os

import numpy as np
import pytest

from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import molnet

expect_bbbp_lengths = [1631, 203, 205]
expect_clearance_lengths = [669, 83, 85]

def test_get_molnet_filepath_without_download():
    filepath = molnet.get_molnet_filepath('bbbp', download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = molnet.get_molnet_filepath('bbbp', download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)

@pytest.mark.slow
def test_get_molnet_filepath_with_download():
    filepath = molnet.get_molnet_filepath('bbbp', download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = molnet.get_molnet_filepath('bbbp', download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)

# bbbp is one of classification task dataset
@pytest.mark.slow
def test_get_molnet_bbbp_dataset():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('bbbp', preprocessor=pp)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = np.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == np.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == np.int32
        assert len(dataset) == expect_bbbp_lengths[i]

@pytest.mark.slow
def test_get_molnet_bbbp_dataset_with_smiles():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('bbbp', preprocessor=pp,
                                         return_smiles=True)

    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    smileses = datasets['smiles']
    datasets = datasets['dataset']
    assert len(smileses) == 3
    assert len(datasets) == 3

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = np.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == np.int32
        # (atom from, atom to) or (edge_type, atom from, atom to) assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == np.int32
        assert len(dataset) == expect_bbbp_lengths[i]
        assert len(smileses[i]) == expect_bbbp_lengths[i]


# clearance is one of classification task dataset
@pytest.mark.slow
def test_get_molnet_clearance_dataset():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('clearance', preprocessor=pp)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = np.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == np.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == np.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_clearance_lengths[i]


@pytest.mark.slow
def test_get_molnet_clearance_dataset():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('clearance', preprocessor=pp,
                                         return_smiles=True)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    smileses = datasets['smiles']
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert len(smileses) == 3

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = np.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == np.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == np.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_clearance_lengths[i]
        assert len(smileses[i]) == expect_clearance_lengths[i]


if __name__ == '__main__':
    args = [__file__, '-v', '-s']
    pytest.main(args=args)
