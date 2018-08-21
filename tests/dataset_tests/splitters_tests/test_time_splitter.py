import os

import numpy
import pytest

from chainer_chemistry.dataset.splitters import time_splitter
from chainer_chemistry.dataset.splitters.time_splitter import TimeSplitter
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture()
def dataset():
    a = numpy.random.random((10, 10))
    b = numpy.random.random((10, 8))
    c = numpy.random.random((10, 1))
    return NumpyTupleDataset(a, b, c)


def test_get_year_table_filepath():
    filepath = time_splitter.get_year_table_filepath(
        download_if_no_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)

    filepath = time_splitter.get_year_table_filepath(download_if_no_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)


def test_split(dataset):
    splitter = TimeSplitter()
    train_ind, valid_ind, test_ind = splitter._split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1

    train_ind, valid_ind, test_ind = splitter._split(
        dataset, frac_train=0.5, frac_valid=0.3, frac_test=0.2)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 5
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 2


def test_split_fail(dataset):
    splitter = TimeSplitter()
    with pytest.raises(AssertionError):
        train_ind, valid_ind, test_ind = splitter._split(
            dataset, frac_train=0.4, frac_valid=0.3, frac_test=0.2)


def test_train_valid_test_split(dataset):
    splitter = TimeSplitter()
    train_ind, valid_ind, test_ind = splitter.train_valid_test_split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 8
    assert valid_ind.shape[0] == 1
    assert test_ind.shape[0] == 1


def test_train_valid_test_split_return_dataset(dataset):
    splitter = TimeSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1


def test_train_valid_split(dataset):
    splitter = TimeSplitter()
    train_ind, valid_ind = splitter.train_valid_split(dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 9
    assert valid_ind.shape[0] == 1


def test_train_split_return_dataset(dataset):
    splitter = TimeSplitter()
    train, valid = splitter.train_valid_split(dataset, return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 9
    assert len(valid) == 1
