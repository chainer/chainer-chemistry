from typing import Tuple  # NOQA

import numpy
import pytest

from chainer import cuda
from chainer import gradient_check

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.mpnn_readout import MPNNReadout
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
in_channels = 7
out_dim = 4
batch_size = 2


@pytest.fixture
def readout():
    # type: () -> MPNNReadout
    return MPNNReadout(out_dim=out_dim, in_channels=in_channels, n_layers=2)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.uniform(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size,
                                      in_channels)).astype('f')
    y_grad = numpy.random.uniform(-1, 1, (batch_size, out_dim)).astype('f')
    return atom_data, y_grad


def check_forward(readout, atom_data):
    # type: (MPNNReadout, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(readout(atom_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_foward_cpu(readout, data):
    # type: (MPNNReadout, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = data[0]
    check_forward(readout, atom_data)


@pytest.mark.gpu
def test_foward_gpu(readout, data):
    # type: (MPNNReadout, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = cuda.to_gpu(data[0])
    readout.to_gpu()
    check_forward(readout, atom_data)


def test_backward_cpu(readout, data):
    # type: (MPNNReadout, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data, y_grad = data
    gradient_check.check_backward(
        readout, atom_data, y_grad, atol=1e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(readout, data):
    # type: (MPNNReadout, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data, y_grad = map(cuda.to_gpu, data)
    readout.to_gpu()
    gradient_check.check_backward(
        readout, atom_data, y_grad, atol=1e-1, rtol=1e-1)


def test_foward_cpu_graph_invariant(readout, data):
    # type: (MPNNReadout, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = data[0]
    y_actual = cuda.to_cpu(readout(atom_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    permute_y_actual = cuda.to_cpu(readout(permute_atom_data).data)
    numpy.testing.assert_allclose(
        y_actual, permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
