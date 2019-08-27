import chainer
import numpy
import pytest
from chainer import links
import chainerx
from chainer.iterators import SerialIterator

from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.models import Regressor
from chainer_chemistry.utils import run_train


input_dim = 5
output_dim = 7
train_data_size = 9
valid_data_size = 8
batch_size = 4


@pytest.fixture
def model():
    return Regressor(links.Linear(None, output_dim))


@pytest.fixture
def train_data():
    x = numpy.random.uniform(
        0, 1, (train_data_size, input_dim)).astype(numpy.float32)
    y = numpy.random.uniform(
        0, 1, (train_data_size, output_dim)).astype(numpy.float32)
    return NumpyTupleDataset(x, y)


@pytest.fixture
def valid_data():
    x = numpy.random.uniform(
        0, 1, (valid_data_size, input_dim)).astype(numpy.float32)
    y = numpy.random.uniform(
        0, 1, (valid_data_size, output_dim)).astype(numpy.float32)
    return NumpyTupleDataset(x, y)


def test_run_train_cpu(model, train_data, valid_data):
    run_train(model, train_data, valid=valid_data, epoch=1, batch_size=8)


def test_run_train_cpu_iterator(model, train_data, valid_data):
    train_iter = SerialIterator(train_data, batch_size=4)
    valid_iter = SerialIterator(valid_data, batch_size=4,
                                shuffle=False, repeat=False)
    run_train(model, train_iter, valid=valid_iter, epoch=1, batch_size=8,
              extensions_list=[lambda t: None])


def test_run_train_invalid(model, train_data):
    with pytest.raises(ValueError):
        run_train(model, train_data, optimizer=1)


@pytest.mark.gpu
def test_run_train_gpu(model, train_data, valid_data):
    device = 0
    model.to_gpu(device)
    run_train(model, train_data, valid=valid_data, epoch=1, batch_size=8,
              device=device)


@pytest.mark.skipif(not chainerx.is_available(),
                    reason='chainerx is not available')
def test_run_train_chainerx_native(model, train_data, valid_data):
    device = chainer.get_device('native')
    model.to_device(device)
    run_train(model, train_data, valid=valid_data, epoch=1, batch_size=8,
              device=device)


@pytest.mark.gpu
@pytest.mark.skipif(not chainerx.is_available(),
                    reason='chainerx is not available')
def test_run_train_chainerx_cuda0(model, train_data, valid_data):
    device = chainer.get_device('cuda:0')
    model.to_device(device)
    run_train(model, train_data, valid=valid_data, epoch=1, batch_size=8,
              device=device)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
