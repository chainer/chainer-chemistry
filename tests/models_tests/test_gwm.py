import chainer
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.dataset.preprocessors.common import construct_supernode_feature  # NOQA
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.models.gwm import GWM
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
hidden_dim = 4
supernode_dim = 7
batch_size = 2
num_edge_type = 2


@pytest.fixture
def gwm():
    return GWM(hidden_dim=hidden_dim, hidden_dim_super=supernode_dim)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
        ).astype('i')
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_dim)).astype('f')
    supernode = numpy.random.uniform(0, 1, (batch_size, supernode_dim))\
        .astype('f')

    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=hidden_dim)
    embed_atom_data = embed(atom_data).data
    return embed_atom_data, supernode, y_grad


def check_forward(gwm, embed_atom_data, supernode):
    gwm.GRU_local.reset_state()
    gwm.GRU_super.reset_state()
    h_actual, g_actual = gwm(embed_atom_data, embed_atom_data, supernode)
    assert h_actual.array.shape == (batch_size, atom_size, hidden_dim)
    assert g_actual.array.shape == (batch_size, supernode_dim)


def test_forward_cpu(gwm, data):
    embed_atom_data, supernode = data[:2]
    check_forward(gwm, embed_atom_data, supernode)


# @pytest.mark.gpu
# def test_forward_gpu(update, data):
#     atom_data, adj_data = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
#     update.to_gpu()
#     check_forward(update, atom_data, adj_data)
#
#
# def check_backward(update, atom_data, adj_data, y_grad):
#     """Check gradient of GGNNUpdate.
#
#     This function is different from other backward tests.
#     Because of GRU, reset_state method has to be called explicitly
#     before gradient calculation.
#
#     Args:
#         update (callable):
#         atom_data (numpy.ndarray):
#         adj_data (numpy.ndarray):
#         y_grad (numpy.ndarray):
#     """
#     atom = chainer.Variable(atom_data)
#     adj = chainer.Variable(adj_data)
#     update.reset_state()
#     y = update(atom, adj)
#     y.grad = y_grad
#     y.backward()
#
#     def f():
#         update.reset_state()
#         return update(atom_data, adj_data).data,
#
#     gx, = gradient_check.numerical_grad(f, (atom.data, ), (y.grad, ))
#     numpy.testing.assert_allclose(
#         cuda.to_cpu(gx), cuda.to_cpu(atom.grad), atol=1e-3, rtol=1e-3)
#
#
# def test_backward_cpu(update, data):
#     check_backward(update, *data)
#
#
# @pytest.mark.gpu
# def test_backward_gpu(update, data):
#     update.to_gpu()
#     check_backward(update, *map(cuda.to_gpu, data))
#
#
# def test_forward_cpu_graph_invariant(update, data):
#     permutation_index = numpy.random.permutation(atom_size)
#     atom_data, adj_data = data[:2]
#     update.reset_state()
#     y_actual = cuda.to_cpu(update(atom_data, adj_data).data)
#
#     permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
#     permute_adj_data = permute_adj(adj_data, permutation_index)
#     update.reset_state()
#     permute_y_actual = cuda.to_cpu(update(
#         permute_atom_data, permute_adj_data).data)
#     numpy.testing.assert_allclose(
#         permute_node(y_actual, permutation_index, axis=1),
#         permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
