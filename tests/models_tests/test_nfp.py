from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainerchem.config import MAX_ATOMIC_NUM
from chainerchem.models.nfp import NFP

atom_size = 5
out_dim = 4
batch_size = 2


@pytest.fixture
def model():
    return NFP(out_dim=out_dim)


@pytest.fixture
def data():
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    # adj_data = numpy.random.randint(
    #     0, high=2, size=(batch_size, atom_size, atom_size)
    # ).astype(numpy.float32)

    # adj_data is symmetric matrix
    adj_data = numpy.random.uniform(
        0, high=1, size=(batch_size, atom_size, atom_size)
    ).astype(numpy.float32)
    adj_data = adj_data + adj_data.swapaxes(-1, -2)
    adj_data = (adj_data > 1.5).astype(numpy.float32)

    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)
    return atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    atom_data, adj_data = data[0], data[1]
    check_forward(model, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    atom_data, adj_data = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
    model.to_gpu()
    check_forward(model, atom_data, adj_data)


# TODO(nakago): check why tolerance is high
def test_backward_cpu(model, data):
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e0, rtol=1e0)


# TODO(nakago): check why tolerance is high
@pytest.mark.gpu
def test_backward_gpu(model, data):
    atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e0, rtol=1e0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
