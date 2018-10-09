import os

import numpy
import pandas
import pytest

from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import molnet
from chainer_chemistry.datasets import NumpyTupleDataset

expect_bbbp_lengths = [1633, 203, 203]
expect_bbbp_lengths2 = [1021, 611, 407]
expect_clearance_lengths = [669, 83, 85]
expect_pdbbind_lengths = [134, 16, 18]
expect_featurized_pdbbind_lengths = [151, 18, 20]
expect_qm7_lengths = [5468, 683, 683]


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


def test_get_grid_featurized_pdbbind_dataset():
    # Test core dataset
    dataset = molnet.get_grid_featurized_pdbbind_dataset('core')
    assert isinstance(dataset, NumpyTupleDataset)
    x, y = dataset.get_datasets()
    assert x.shape == (189, 2052)
    assert x.dtype == numpy.int32
    assert y.shape == (189, 1)
    assert y.dtype == numpy.float32

    # Test full dataset
    dataset = molnet.get_grid_featurized_pdbbind_dataset('full')
    assert isinstance(dataset, NumpyTupleDataset)
    x, y = dataset.get_datasets()
    assert x.shape == (11303, 2052)
    assert x.dtype == numpy.int32
    assert y.shape == (11303, 1)
    assert y.dtype == numpy.float32

    # Test refined dataset
    dataset = molnet.get_grid_featurized_pdbbind_dataset('refined')
    assert isinstance(dataset, NumpyTupleDataset)
    x, y = dataset.get_datasets()
    assert x.shape == (3568, 2052)
    assert x.dtype == numpy.int32
    assert y.shape == (3568, 1)
    assert y.dtype == numpy.float32


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
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.int32
        assert len(dataset) == expect_bbbp_lengths[i]


# bbbp is one of classification task dataset
@pytest.mark.slow
def test_get_molnet_bbbp_dataset_change_split_ratio():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('bbbp', preprocessor=pp,
                                         frac_train=0.5, frac_valid=0.3,
                                         frac_test=0.2)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.int32
        assert len(dataset) == expect_bbbp_lengths2[i]


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
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to) assert label.ndim == 1 # NOQA
        assert label.shape[0] == 1
        assert label.dtype == numpy.int32
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
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_clearance_lengths[i]


@pytest.mark.slow
def test_get_molnet_clearance_dataset_with_return_smiles_enabled():
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
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_clearance_lengths[i]
        assert len(smileses[i]) == expect_clearance_lengths[i]


@pytest.mark.slow
def test_get_molnet_pdbbind_dataset():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    time_list = numpy.random.randint(1000, size=168).tolist()
    datasets = molnet.get_molnet_dataset('pdbbind_smiles', preprocessor=pp,
                                         pdbbind_subset='core',
                                         time_list=time_list, split='random')
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    assert 'pdb_id' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_pdbbind_lengths[i]


@pytest.mark.slow
def test_get_molnet_pdbbind_dataset_with_pdb_id():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    time_list = numpy.random.randint(1000, size=168).tolist()
    datasets = molnet.get_molnet_dataset('pdbbind_smiles', preprocessor=pp,
                                         pdbbind_subset='core',
                                         return_pdb_id=True,
                                         time_list=time_list, split='random')
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    assert 'pdb_id' in datasets.keys()
    pdb_ids = datasets['pdb_id']
    datasets = datasets['dataset']
    assert len(pdb_ids) == 3
    assert len(datasets) == 3

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert label.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --Test number of dataset ---
        assert len(dataset) == expect_pdbbind_lengths[i]
        assert len(pdb_ids[i]) == expect_pdbbind_lengths[i]


@pytest.mark.slow
def test_get_molnet_grid_featurized_pdbbind_dataset():
    # test default behavioer
    datasets = molnet.get_molnet_dataset('pdbbind_grid', pdbbind_subset='core',
                                         split='random')
    assert 'dataset' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_featurized_pdbbind_lengths[i]


# For qm7 dataset, stratified splitting is recommended.
@pytest.mark.slow
def test_get_molnet_qm7_dataset():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('qm7', preprocessor=pp)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_qm7_lengths[i]


# For qm7 dataset, stratified splitting is recommended.
@pytest.mark.slow
def test_get_molnet_qm7_dataset_with_smiles():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    datasets = molnet.get_molnet_dataset('qm7', preprocessor=pp,
                                         return_smiles=True)
    assert 'smiles' in datasets.keys()
    assert 'dataset' in datasets.keys()
    smileses = datasets['smiles']
    datasets = datasets['dataset']
    assert len(datasets) == 3
    assert len(smileses) == 3
    assert type(datasets[0]) == NumpyTupleDataset
    assert type(datasets[1]) == NumpyTupleDataset
    assert type(datasets[2]) == NumpyTupleDataset

    # Test each train, valid and test dataset
    for i, dataset in enumerate(datasets):
        # --- Test dataset is correctly obtained ---
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert label.ndim == 1
        assert label.shape[0] == 1
        assert label.dtype == numpy.float32

        # --- Test number of dataset ---
        assert len(dataset) == expect_qm7_lengths[i]
        assert len(smileses[i]) == expect_qm7_lengths[i]


def test_get_molnet_bbbp_dataframe():
    datasets = molnet.get_molnet_dataframe('bbbp')
    assert isinstance(datasets, pandas.DataFrame)
    assert len(datasets) == 2050


if __name__ == '__main__':
    args = [__file__, '-v', '-s']
    pytest.main(args=args)
