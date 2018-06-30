import numpy
import pytest

from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter
from chainer_chemistry.datasets import NumpyTupleDataset


@pytest.fixture
def dataset():
    a = numpy.random.random((10, 10))
    b = numpy.random.random((10, 8))
    c = numpy.random.random((10, 1))
    return NumpyTupleDataset(a, b, c)


@pytest.fixture
def ndarray_dataset():
    a = numpy.random.random((10, 10))
    return a


def test_split(dataset):
    splitter = RandomSplitter()
    train_ind, valid_ind, test_ind = splitter._split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1

    train_ind, valid_ind, test_ind = splitter._split(dataset,
                                                     frac_train=0.5,
                                                     frac_valid=0.3,
                                                     frac_test=0.2)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 5
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 2


def test_split_fix_seed(dataset):
    splitter = RandomSplitter()
    train_ind1, valid_ind1, test_ind1 = splitter._split(dataset, seed=44)
    train_ind2, valid_ind2, test_ind2 = splitter._split(dataset, seed=44)

    assert numpy.array_equal(train_ind1, train_ind2)
    assert numpy.array_equal(valid_ind1, valid_ind2)
    assert numpy.array_equal(test_ind1, test_ind2)


def test_split_fail(dataset):
    splitter = RandomSplitter()
    with pytest.raises(AssertionError):
        train_ind, valid_ind, test_ind = splitter._split(dataset,
                                                         frac_train=0.4,
                                                         frac_valid=0.3,
                                                         frac_test=0.2)


def test_train_valid_test_split(dataset):
    splitter = RandomSplitter()
    train_ind, valid_ind, test_ind = splitter.train_valid_test_split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1


def test_train_valid_test_split_return_dataset(dataset):
    splitter = RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(dataset,
                                                         return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1


def test_train_valid_test_split_ndarray_return_dataset(ndarray_dataset):
    splitter = RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(ndarray_dataset,
                                                         return_index=False)
    assert type(train) == numpy.ndarray
    assert type(valid) == numpy.ndarray
    assert type(test) == numpy.ndarray
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1


def test_train_valid_split(dataset):
    splitter = RandomSplitter()
    train_ind, valid_ind = splitter.train_valid_split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 9
    assert valid_ind.shape[0] == 1


def test_train_valid_split_return_dataset(dataset):
    splitter = RandomSplitter()
    train, valid = splitter.train_valid_split(dataset, return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 9
    assert len(valid) == 1
