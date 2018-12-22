from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.links.connection.graph_mlp import GraphMLP  # NOQA

in_size = 3
atom_size = 5
out_size = 4
channels = [16, out_size]
batch_size = 2


@pytest.fixture
def model():
    l = GraphMLP(channels, in_channels=in_size)
    l.cleargrads()
    return l


@pytest.fixture
def data():
    x_data = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, in_size)).astype(numpy.float32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, out_size)).astype(numpy.float32)
    return x_data, y_grad


def test_forward_cpu(model, data):
    # only testing shape for now...
    x_data = data[0]
    y_actual = model(x_data)
    assert y_actual.shape == (batch_size, atom_size, out_size)
    assert len(model.layers) == len(channels)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    x_data = cuda.to_gpu(data[0])
    model.to_gpu()
    y_actual = model(x_data)
    assert y_actual.shape == (batch_size, atom_size, out_size)
    assert len(model.layers) == len(channels)


def test_backward_cpu(model, data):
    x_data, y_grad = data
    gradient_check.check_backward(model, x_data, y_grad, list(model.params()),
                                  atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    x_data, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(model, x_data, y_grad, list(model.params()),
                                  atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
