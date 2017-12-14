from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.weavenet import WeaveNet

atom_size = 5
weave_channels = [50, 50]
batch_size = 2
atom_feature_dim = 23
pair_feature_dim = 10
out_dim = weave_channels[-1]


@pytest.fixture
def model():
    return WeaveNet(weave_channels=weave_channels, n_atom=atom_size)


@pytest.fixture
def model_processed():
    """model to test `atom_data_processed` input"""
    return WeaveNet(weave_channels=weave_channels, n_atom=atom_size)


@pytest.fixture
def data():
    atom_data_processed = numpy.random.uniform(
        0, high=1, size=(batch_size, atom_size, atom_feature_dim)
    ).astype(numpy.float32)

    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    # adj_data is symmetric along pair of atoms
    adj_data = numpy.random.uniform(
        0, high=1, size=(batch_size, pair_feature_dim, atom_size, atom_size)
    ).astype(numpy.float32)
    adj_data = adj_data + adj_data.swapaxes(-1, -2)
    adj_data = adj_data.transpose((0, 3, 2, 1)).reshape(
        batch_size, atom_size * atom_size, pair_feature_dim
    ).astype(numpy.float32)

    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)
    return atom_data_processed, atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    print('y_actual', y_actual.shape)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, model_processed, data):
    atom_data_processed, atom_data, adj_data = data[0:3]
    check_forward(model, atom_data, adj_data)
    check_forward(model_processed, atom_data_processed, adj_data)


@pytest.mark.gpu
def test_forward_gpu(model, model_processed, data):
    atom_data_processed, atom_data, adj_data = \
        [cuda.to_gpu(d) for d in data[0:3]]
    model.to_gpu()
    model_processed.to_gpu()
    check_forward(model, atom_data, adj_data)
    check_forward(model_processed, atom_data_processed, adj_data)


def test_backward_cpu(model, model_processed, data):
    atom_data_processed, atom_data, adj_data, y_grad = data
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e-1, rtol=1e-1)
    gradient_check.check_backward(model_processed, (atom_data_processed, adj_data),
                                  y_grad, atol=1e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(model, model_processed, data):
    atom_data_processed, atom_data, adj_data, y_grad = \
        [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    model_processed.to_gpu()
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e-1, rtol=1e-1)
    gradient_check.check_backward(model_processed, (atom_data_processed, adj_data),
                                  y_grad, atol=1e-1, rtol=1e-1)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
