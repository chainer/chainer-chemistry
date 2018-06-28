import numpy
import pytest

from chainer_chemistry.dataset.splitters.stratified_splitter import StratifiedSplitter # NOQA
from chainer_chemistry.datasets import NumpyTupleDataset


@pytest.fixture
def cls_dataset():
    a = numpy.random.random((30, 10))
    b = numpy.random.random((30, 8))
    # c = numpy.random.randint(2, size=(30, 3))
    c = numpy.concatenate([numpy.zeros(20), numpy.ones(10)]).astype(numpy.int)
    return NumpyTupleDataset(a, b, c)


def test_classification_split(cls_dataset):
    splitter = StratifiedSplitter()
    train_ind, valid_ind, test_ind = splitter._split(cls_dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 24
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 3

    train_ind, valid_ind, test_ind = splitter._split(cls_dataset,
                                                     frac_train=0.5,
                                                     frac_valid=0.3,
                                                     frac_test=0.2)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 15
    assert valid_ind.shape[0] == 9
    assert test_ind.shape[0] == 6


def test_train_valid_test_classification_split(cls_dataset):
    splitter = StratifiedSplitter()
    train_ind, valid_ind, test_ind =\
        splitter.train_valid_test_split(cls_dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 24
    assert valid_ind.shape[0] == 3
    assert test_ind.shape[0] == 3


def test_train_valid_test_classification_split_return_dataset(cls_dataset):
    splitter = StratifiedSplitter()
    train, valid, test = splitter.train_valid_test_split(cls_dataset,
                                                         return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert type(test) == NumpyTupleDataset
    assert len(train) == 24
    assert len(valid) == 3
    assert len(test) == 3
    assert (train.features[:, -1] == 1).sum() == 8
    assert (valid.features[:, -1] == 1).sum() == 1
    assert (test.features[:, -1] == 1).sum() == 1


def test_train_valid_split(cls_dataset):
    splitter = StratifiedSplitter()
    train_ind, valid_ind = splitter.train_valid_split(cls_dataset)
    assert type(train_ind) == numpy.ndarray
    assert train_ind.shape[0] == 27
    assert valid_ind.shape[0] == 3


def test_train_valid_split_return_dataset(cls_dataset):
    splitter = StratifiedSplitter()
    train, valid = splitter.train_valid_split(cls_dataset, return_index=False)
    assert type(train) == NumpyTupleDataset
    assert type(valid) == NumpyTupleDataset
    assert len(train) == 27
    assert len(valid) == 3
    assert (train.features[:, -1] == 1).sum() == 9
    assert (valid.features[:, -1] == 1).sum() == 1
