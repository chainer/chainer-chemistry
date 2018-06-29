import os

import numpy
import pytest


from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser # NOQA
from chainer_chemistry.dataset.preprocessors import NFPPreprocessor
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def dataset():
    here = os.path.abspath(os.path.dirname(__file__))
    file_rel_path = '../../../examples/own_dataset/dataset.csv'
    file_abs_path = os.path.join(here, file_rel_path)
    pp = NFPPreprocessor()
    parser = CSVFileParser(pp, smiles_col='SMILES')
    dataset = parser.parse(file_abs_path, return_smiles=True)
    return dataset


def test_split(dataset):
    splitter = ScaffoldSplitter()
    train_ind, valid_ind, test_ind = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'])
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 80
    assert valid_ind.shape[0] == 10
    assert test_ind.shape[0] == 10

    train_ind, valid_ind, test_ind = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'],
        frac_train=0.5, frac_valid=0.3, frac_test=0.2)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 50
    assert valid_ind.shape[0] == 30
    assert test_ind.shape[0] == 20


def test_split_fix_seed(dataset):
    splitter = ScaffoldSplitter()
    train_ind1, valid_ind1, test_ind1 = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'], seed=44)
    train_ind2, valid_ind2, test_ind2 = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'], seed=44)

    assert numpy.array_equal(train_ind1, train_ind2)
    assert numpy.array_equal(valid_ind1, valid_ind2)
    assert numpy.array_equal(test_ind1, test_ind2)


def test_split_fail(dataset):
    splitter = ScaffoldSplitter()
    with pytest.raises(AssertionError):
        train_ind, valid_ind, test_ind = splitter._split(
            dataset=dataset['dataset'], smiles_list=dataset['smiles'],
            frac_train=0.4, frac_valid=0.3, frac_test=0.2)


def test_train_valid_test_split(dataset):
    splitter = ScaffoldSplitter()
    train_ind, valid_ind, test_ind = splitter.train_valid_test_split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'])

    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 80
    assert valid_ind.shape[0] == 10
    assert test_ind.shape[0] == 10


def test_train_valid_test_split_return_dataset(dataset):
    splitter = ScaffoldSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'],
        return_index=False)

    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 80
    assert len(valid) == 10
    assert len(test) == 10


def test_train_valid_split(dataset):
    splitter = ScaffoldSplitter()
    train_ind, valid_ind = splitter.train_valid_split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'])

    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 90
    assert valid_ind.shape[0] == 10


def test_train_valid_split_return_dataset(dataset):
    splitter = ScaffoldSplitter()
    train, valid = splitter.train_valid_split(dataset=dataset['dataset'],
                                              smiles_list=dataset['smiles'],
                                              return_index=False)

    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 90
    assert len(valid) == 10
