import numpy
import pandas
import pytest

from chainer_chemistry.dataset.parsers.data_frame_parser import DataFrameParser  # NOQA
from chainer_chemistry.dataset.preprocessors import AtomicNumberPreprocessor
from chainer_chemistry.dataset.splitters.scaffold_splitter import generate_scaffold  # NOQA
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def smiles_list():
    smileses = [
        "CC1=CC2CC(CC1)O2",
        "O=Cc1nccn1C=O",
        "CCC(C)(C)C(O)C=O",
        "C#CCC(C)(CO)OC",
        "Nc1coc(=O)nc1N",
        "CC12C=CC(CCC1)C2",
        "CC12CCC1C2OC=O",
        "CC1C2CC3(COC3)N12",
        "O=C1NC=NC12CC2",
        "C1=CC2CN2CC2NC12",
    ]
    return smileses


@pytest.fixture
def dataset(smiles_list):
    df = pandas.DataFrame(data={'smiles': smiles_list,
                                'value': numpy.random.rand(10)})
    pp = AtomicNumberPreprocessor()
    parser = DataFrameParser(pp, labels='value')
    dataset = parser.parse(df, return_smiles=True)
    return dataset


def test_generate_scaffold():
    smiles = "Nc1coc(=O)nc1N"
    actual = generate_scaffold(smiles)
    expect = 'O=c1nccco1'
    assert actual == expect


def test_split(dataset):
    splitter = ScaffoldSplitter()
    train_ind, valid_ind, test_ind = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'])
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1

    train_ind, valid_ind, test_ind = splitter._split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'],
        frac_train=0.5, frac_valid=0.3, frac_test=0.2)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 5
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 2


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
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1


def test_train_valid_test_split_return_dataset(dataset):
    splitter = ScaffoldSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'],
        return_index=False)

    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1


def test_train_valid_split(dataset):
    splitter = ScaffoldSplitter()
    train_ind, valid_ind = splitter.train_valid_split(
        dataset=dataset['dataset'], smiles_list=dataset['smiles'])

    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 9
    assert valid_ind.shape[0] == 1


def test_train_valid_split_return_dataset(dataset):
    splitter = ScaffoldSplitter()
    train, valid = splitter.train_valid_split(dataset=dataset['dataset'],
                                              smiles_list=dataset['smiles'],
                                              return_index=False)

    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 9
    assert len(valid) == 1
