from typing import Tuple  # NOQA

import chainer  # NOQA
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.update.gin_update import GINUpdate
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
in_channels = 4
hidden_channels = 6
batch_size = 3
num_edge_type = 7


@pytest.fixture
def update():
    # type: () -> GINUpdate
    return GINUpdate(in_channels=in_channels, hidden_channels=hidden_channels,
                     dropout_ratio=0)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)).astype('i')
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, atom_size, atom_size)).astype('f')
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_channels)).astype('f')
    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=in_channels)
    embed_atom_data = embed(atom_data).data
    return embed_atom_data, adj_data, y_grad


# Test Update Function
def check_forward(update, atom_data, adj_data):
    # type: (GINUpdate, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(update(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_channels)


def test_forward_cpu(update, data):
    # type: (GINUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    check_forward(update, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(update, data):
    # type: (GINUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    update.to_gpu()
    check_forward(update, atom_data, adj_data)


def test_backward_cpu(update, data):
    # type: (GINUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        update, (atom_data, adj_data), y_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(update, data):
    # type: (GINUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = map(cuda.to_gpu, data[:3])
    update.to_gpu()
    gradient_check.check_backward(
        update, (atom_data, adj_data), y_grad, atol=1e-3, rtol=1e-3)


def test_forward_cpu_graph_invariant(update, data):
    # type: (GINUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    y_actual = cuda.to_cpu(update(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    permute_y_actual = cuda.to_cpu(
        update(permute_atom_data, permute_adj_data).data)
    numpy.testing.assert_allclose(
        permute_node(y_actual, permutation_index, axis=1),
        permute_y_actual,
        rtol=1e-3,
        atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
