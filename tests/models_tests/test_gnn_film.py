from typing import Tuple  # NOQA

import chainer  # NOQA
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.gnn_film import GNNFiLM
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
out_dim = 4
batch_size = 3
n_edge_types = 5


@pytest.fixture
def model():
    # type: () -> chainer.Chain
    return GNNFiLM(out_dim=out_dim)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)).astype('i')
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, n_edge_types, atom_size, atom_size)
    ).astype('f')
    y_grad = numpy.random.uniform(-1, 1, (batch_size, out_dim)).astype('f')
    return atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    # type: (chainer.Chain, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    # type: (chainer.Chain, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    check_forward(model, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    # type: (chainer.Chain, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    model.to_gpu()
    check_forward(model, atom_data, adj_data)


def test_backward_cpu(model, data):
    # type: (chainer.Chain, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    # type: (chainer.Chain, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = map(cuda.to_gpu, data)
    model.to_gpu()
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-2, rtol=1e-2)


def test_forward_cpu_graph_invariant(model, data):
    # type: (chainer.Chain, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    permute_y_actual = cuda.to_cpu(
        model(permute_atom_data, permute_adj_data).data)
    numpy.testing.assert_allclose(
        y_actual, permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
