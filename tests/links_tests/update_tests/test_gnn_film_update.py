from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.update.gnn_film_update import GNNFiLMUpdate
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
in_channels = 7
hidden_channels = 7
batch_size = 2
n_edge_types = 5


@pytest.fixture
def update():
    return GNNFiLMUpdate(hidden_channels=hidden_channels,
                         n_edge_types=n_edge_types)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype('i')
    adj_data = numpy.random.uniform(
        0, high=2, size=(batch_size, n_edge_types, atom_size, atom_size)
    ).astype('f')
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_channels)).astype('f')

    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=in_channels)
    embed_atom_data = embed(atom_data).data
    adj_data = adj_data
    return embed_atom_data, adj_data, y_grad


# Test Update Function
def check_forward(update, atom_data, adj_data):
    # type: (GNNFiLMUpdate, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(update(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_channels)


def test_forward_cpu(update, data):
    # type: (GNNFiLMUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    check_forward(update, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(update, data):
    # type: (GNNFiLMUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    update.to_gpu()
    check_forward(update, atom_data, adj_data)


def test_backward_cpu(update, data):
    # type: (GNNFiLMUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        update, (atom_data, adj_data), y_grad, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_backward_gpu(update, data):
    # type: (GNNFiLMUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = map(cuda.to_gpu, data[:3])
    update.to_gpu()
    # print(type(adj_data))
    gradient_check.check_backward(
        update, (atom_data, adj_data), y_grad, atol=1e-2, rtol=1e-2)


def test_forward_cpu_graph_invariant(update, data):
    # type: (GNNFiLMUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
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
