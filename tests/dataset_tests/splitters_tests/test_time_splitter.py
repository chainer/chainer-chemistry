import numpy
import pytest

from chainer_chemistry.dataset.splitters.time_splitter import TimeSplitter
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def time_list():
    times = [
        1980,
        1990,
        2010,
        2020,
        2000,
        2050,
        2030,
        2040,
        1960,
        1970
    ]
    return times


@pytest.fixture()
def dataset():
    a = numpy.random.random((10, 10))
    b = numpy.random.random((10, 8))
    c = numpy.random.random((10, 1))
    return NumpyTupleDataset(a, b, c)


def test_split(dataset, time_list):
    splitter = TimeSplitter()
    train_ind, valid_ind, test_ind = splitter._split(
        dataset, time_list=time_list)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1
    assert train_ind.tolist() == [8, 9, 0, 1, 4, 2, 3, 6]
    assert valid_ind.tolist() == [7]
    assert test_ind.tolist() == [5]

    train_ind, valid_ind, test_ind = splitter._split(
        dataset, frac_train=0.5, frac_valid=0.3, frac_test=0.2,
        time_list=time_list)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 5
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 2
    assert train_ind.tolist() == [8, 9, 0, 1, 4]
    assert valid_ind.tolist() == [2, 3, 6]
    assert test_ind.tolist() == [7, 5]


def test_split_fail(dataset, time_list):
    splitter = TimeSplitter()
    with pytest.raises(AssertionError):
        train_ind, valid_ind, test_ind = splitter._split(
            dataset, frac_train=0.4, frac_valid=0.3, frac_test=0.2,
            time_list=time_list)


def test_train_valid_test_split(dataset, time_list):
    splitter = TimeSplitter()
    train_ind, valid_ind, test_ind = splitter.train_valid_test_split(
        dataset, time_list=time_list)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1
    assert train_ind.tolist() == [8, 9, 0, 1, 4, 2, 3, 6]
    assert valid_ind.tolist() == [7]
    assert test_ind.tolist() == [5]


def test_train_valid_test_split_return_dataset(dataset, time_list):
    splitter = TimeSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, return_index=False, time_list=time_list)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1


def test_train_valid_split(dataset, time_list):
    splitter = TimeSplitter()
    train_ind, valid_ind = splitter.train_valid_split(
        dataset, time_list=time_list)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 9
    assert valid_ind.shape[0] == 1
    assert train_ind.tolist() == [8, 9, 0, 1, 4, 2, 3, 6, 7]
    assert valid_ind.tolist() == [5]


def test_train_split_return_dataset(dataset, time_list):
    splitter = TimeSplitter()
    train, valid = splitter.train_valid_split(
        dataset, return_index=False, time_list=time_list)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 9
    assert len(valid) == 1
