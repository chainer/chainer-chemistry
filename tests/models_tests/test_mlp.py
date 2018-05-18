from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.models.mlp import MLP

batch_size = 2
hidden_dim = 16
out_dim = 4


@pytest.fixture
def model():
    return MLP(out_dim=out_dim)


@pytest.fixture
def data():
    hidden = numpy.random.rand(batch_size, hidden_dim).astype(numpy.float32)
    y_grad = numpy.random.uniform(-1, 1, (batch_size, out_dim)).astype(
        numpy.float32)
    return hidden, y_grad


def check_forward(model, data):
    y_actual = cuda.to_cpu(model(data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    check_forward(model, data[0])


@pytest.mark.gpu
def test_forward_gpu(model, data):
    model.to_gpu()
    check_forward(model, cuda.to_gpu(data[0]))


def test_mlp_assert_raises():
    with pytest.raises(ValueError):
        MLP(out_dim=out_dim, n_layers=-1)


def test_backward_cpu(model, data):
    hidden, y_grad = data
    gradient_check.check_backward(model, hidden, y_grad, atol=1e0, rtol=1e0)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    hidden, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(model, hidden, y_grad, atol=1e0, rtol=1e0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
