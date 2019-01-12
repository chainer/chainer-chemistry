from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.update.schnet_update import SchNetUpdate
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
hidden_dim = 4
batch_size = 2


@pytest.fixture
def update():
    return SchNetUpdate(hidden_dim=hidden_dim)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype('i')
    # symmetric matrix
    dist_data = numpy.random.uniform(
        0, high=30, size=(batch_size, atom_size, atom_size)).astype('f')
    dist_data = (dist_data + dist_data.swapaxes(-1, -2)) / 2.

    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_dim)).astype('f')
    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=hidden_dim)
    embed_atom_data = embed(atom_data).data
    return embed_atom_data, dist_data, y_grad


def check_forward(update, atom_data, dist_data):
    y_actual = cuda.to_cpu(update(atom_data, dist_data).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_dim)


def test_forward_cpu(update, data):
    atom_data, dist_data = data[:2]
    check_forward(update, atom_data, dist_data)


@pytest.mark.gpu
def test_forward_gpu(update, data):
    atom_data, dist_data = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
    update.to_gpu()
    check_forward(update, atom_data, dist_data)


def test_backward_cpu(update, data):
    atom_data, dist_data, y_grad = data
    gradient_check.check_backward(
        update, (atom_data, dist_data), y_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(update, data):
    atom_data, dist_data, y_grad = map(cuda.to_gpu, data)
    update.to_gpu()
    gradient_check.check_backward(
        update, (atom_data, dist_data), y_grad, atol=1e-3, rtol=1e-3)


def test_forward_cpu_graph_invariant(update, data):
    atom_data, dist_data = data[:2]
    y_actual = cuda.to_cpu(update(atom_data, dist_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    permute_dist_data = permute_adj(dist_data, permutation_index)
    permute_y_actual = cuda.to_cpu(update(
        permute_atom_data, permute_dist_data).data)
    numpy.testing.assert_allclose(
        permute_node(y_actual, permutation_index, axis=1),
        permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
