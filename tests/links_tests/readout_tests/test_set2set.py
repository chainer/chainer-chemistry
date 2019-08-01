from typing import Tuple  # NOQA

import chainer
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.set2set import Set2Set
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
in_channels = 7
batch_size = 2


@pytest.fixture
def readout():
    # type: () -> Set2Set
    return Set2Set(in_channels=in_channels, n_layers=2)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.uniform(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size,
                                      in_channels)).astype('f')
    y_grad = numpy.random.uniform(-1, 1,
                                  (batch_size, in_channels * 2)).astype('f')
    return atom_data, y_grad


def check_forward(readout, atom_data):
    # type: (Set2Set, numpy.ndarray) -> None
    readout.reset_state()
    y_actual = cuda.to_cpu(readout(atom_data).data)
    assert y_actual.shape == (batch_size, in_channels * 2)


def test_forward_cpu(readout, data):
    # type: (Set2Set, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = data[0]
    check_forward(readout, atom_data)


@pytest.mark.gpu
def test_forward_gpu(readout, data):
    # type: (Set2Set, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = cuda.to_gpu(data[0])
    readout.to_gpu()
    check_forward(readout, atom_data)


def check_backward(readout, atom_data, y_grad):
    # type: (Set2Set, numpy.ndarray, numpy.ndarray) -> None
    """Check gradient of Set2Set.

    This function is different from other backward tests.
    Because of LSTM, reset_state method has to be called explicitly
    before gradient calculation.

    Args:
        readout:
        atom_data:
        y_grad:
    """
    atom = chainer.Variable(atom_data)
    readout.reset_state()
    y = readout(atom)
    y.grad = y_grad
    y.backward()

    def f():
        readout.reset_state()
        return readout(atom_data).data,

    gx, = gradient_check.numerical_grad(f, (atom.data, ), (y.grad, ))
    numpy.testing.assert_allclose(cuda.to_cpu(gx), cuda.to_cpu(atom.grad),
                                  atol=1e-2, rtol=1e-2)


def test_backward_cpu(readout, data):
    # type: (Set2Set, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    check_backward(readout, *data)


@pytest.mark.gpu
def test_backward_gpu(readout, data):
    # type: (Set2Set, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data, y_grad = [cuda.to_gpu(d) for d in data]
    readout.to_gpu()
    check_backward(readout, atom_data, y_grad)


def test_forward_cpu_graph_invariant(readout, data):
    # type: (Set2Set, Tuple[numpy.ndarray, numpy.ndarray]) -> None
    atom_data = data[0]
    readout.reset_state()
    y_actual = cuda.to_cpu(readout(atom_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    readout.reset_state()
    permute_y_actual = cuda.to_cpu(readout(permute_atom_data).data)
    numpy.testing.assert_allclose(
        y_actual, permute_y_actual, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
