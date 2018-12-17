from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import NFPUpdate
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node


atom_size = 5
hidden_channels = 4
batch_size = 2
num_degree_type = 7


@pytest.fixture
def update():
    return NFPUpdate(in_channels=hidden_channels, out_channels=hidden_channels)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)).astype('i')
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, atom_size, atom_size)).astype('f')
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_channels)).astype('f')

    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=hidden_channels)
    embed_atom_data = embed(atom_data).data
    degree_mat = numpy.sum(adj_data, axis=1)
    deg_conds = numpy.array([numpy.broadcast_to(
        ((degree_mat - degree) == 0)[:, :, None], embed_atom_data.shape)
        for degree in range(1, num_degree_type + 1)])
    return embed_atom_data, adj_data, deg_conds, y_grad


def check_forward(update, atom_data, adj_data, deg_conds):
    y_actual = cuda.to_cpu(update(atom_data, adj_data, deg_conds).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_channels)


def test_forward_cpu(update, data):
    atom_data, adj_data, deg_conds = data[:3]
    check_forward(update, atom_data, adj_data, deg_conds)


@pytest.mark.gpu
def test_forward_gpu(update, data):
    atom_data, adj_data, deg_conds = map(cuda.to_gpu, data[:3])
    update.to_gpu()
    check_forward(update, atom_data, adj_data, deg_conds)


def test_backward_cpu(update, data):
    atom_data, adj_data, deg_conds, y_grad = data
    gradient_check.check_backward(
        update, (atom_data, adj_data, deg_conds), y_grad, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(update, data):
    atom_data, adj_data, deg_conds, y_grad = map(cuda.to_gpu, data)
    update.to_gpu()
    gradient_check.check_backward(
        update, (atom_data, adj_data, deg_conds), y_grad, atol=1e-3, rtol=1e-3)


def test_forward_cpu_graph_invariant(update, data):
    atom_data, adj_data, deg_conds = data[:3]
    y_actual = cuda.to_cpu(update(atom_data, adj_data, deg_conds).data)

    permutation_index = numpy.random.permutation(atom_size)
    # atom_data: (batch_size, atom_size, hidden_channels)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    # deg_conds: (num_degree_type, batch_size, atom_size, hidden_channels)
    permute_deg_conds = permute_node(deg_conds, permutation_index, axis=2)
    permute_y_actual = cuda.to_cpu(update(
        permute_atom_data, permute_adj_data, permute_deg_conds).data)
    numpy.testing.assert_allclose(
        permute_node(y_actual, permutation_index, axis=1), permute_y_actual,
        rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
