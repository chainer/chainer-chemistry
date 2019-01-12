from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.schnet_readout import SchNetReadout
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
hidden_dim = 7
out_dim = 4
batch_size = 2


@pytest.fixture
def readout():
    return SchNetReadout(out_dim=out_dim, hidden_dim=hidden_dim)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.uniform(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size, hidden_dim)
    ).astype('f')
    y_grad = numpy.random.uniform(-1, 1, (batch_size, out_dim)).astype('f')
    return atom_data, y_grad


def check_forward(readout, atom_data):
    y_actual = cuda.to_cpu(readout(atom_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(readout, data):
    atom_data = data[0]
    check_forward(readout, atom_data)


@pytest.mark.gpu
def test_forward_gpu(readout, data):
    atom_data = cuda.to_gpu(data[0])
    readout.to_gpu()
    check_forward(readout, atom_data)


def test_backward_cpu(readout, data):
    atom_data, y_grad = data
    gradient_check.check_backward(
        readout, atom_data, y_grad, atol=1e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(readout, data):
    atom_data, y_grad = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
    readout.to_gpu()
    gradient_check.check_backward(
        readout, atom_data, y_grad, atol=1e-1, rtol=1e-1)


def test_forward_cpu_graph_invariant(readout, data):
    atom_data = data[0]
    y_actual = cuda.to_cpu(readout(atom_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    permute_y_actual = cuda.to_cpu(readout(permute_atom_data).data)
    numpy.testing.assert_allclose(
        y_actual, permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
