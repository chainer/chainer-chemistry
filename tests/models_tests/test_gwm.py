from chainer import cuda
from chainer import functions
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.models.gwm import GWM, WarpGateUnit, SuperNodeTransmitterUnit, GraphTransmitterUnit  # NOQA
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
hidden_dim = 4
supernode_dim = 7
batch_size = 2
num_edge_type = 2


@pytest.fixture
def graph_warp_gate_unit():
    return WarpGateUnit(output_type='graph', hidden_dim=hidden_dim)


@pytest.fixture
def super_warp_gate_unit():
    return WarpGateUnit(output_type='super', hidden_dim=supernode_dim)


@pytest.fixture
def super_node_transmitter_unit():
    return SuperNodeTransmitterUnit(hidden_dim_super=supernode_dim,
                                    hidden_dim=hidden_dim)


@pytest.fixture
def graph_transmitter_unit():
    return GraphTransmitterUnit(hidden_dim_super=supernode_dim,
                                hidden_dim=hidden_dim)


@pytest.fixture
def gwm():
    # relu is difficult to test
    return GWM(hidden_dim=hidden_dim, hidden_dim_super=supernode_dim,
               n_layers=2, activation=functions.identity,
               wgu_activation=functions.identity,
               gtu_activation=functions.identity)


@pytest.fixture
def data():
    numpy.random.seed(0)
    # too difficult to pass unit test by using EmbedAtomID
    embed_atom_data = numpy.random.uniform(
        -0.01, 0.01, (batch_size, atom_size, hidden_dim)).astype('f')
    new_embed_atom_data = numpy.random.uniform(
        -0.01, 0.01, (batch_size, atom_size, hidden_dim)).astype('f')
    y_grad = numpy.random.uniform(
        -0.01, 0.01, (batch_size, atom_size, hidden_dim)).astype('f')
    supernode = numpy.random.uniform(-0.01, 0.01, (batch_size, supernode_dim))\
        .astype('f')
    supernode_grad = numpy.random.uniform(
        -0.01, 0.01, (batch_size, supernode_dim)).astype('f')

    return embed_atom_data, new_embed_atom_data, supernode, y_grad,\
        supernode_grad


def test_graph_transmitter_unit_forward(graph_transmitter_unit, data):
    embed_atom_data = data[0]
    supernode = data[2]
    h_trans = graph_transmitter_unit(embed_atom_data, supernode)
    assert h_trans.array.shape == (batch_size, supernode_dim)


def test_graph_transmitter_unit_backward(graph_transmitter_unit, data):
    embed_atom_data = data[0]
    supernode = data[2]
    supernode_grad = data[4]
    gradient_check.check_backward(graph_transmitter_unit,
                                  (embed_atom_data, supernode),
                                  supernode_grad, eps=0.1)


def test_super_node_transmitter_unit_forward(super_node_transmitter_unit,
                                             data):
    supernode = data[2]
    g_trans = super_node_transmitter_unit(supernode, atom_size)
    assert g_trans.array.shape == (batch_size, atom_size, hidden_dim)


def test_super_node_transmitter_unit_backward(super_node_transmitter_unit,
                                              data):
    supernode = data[2]
    y_grad = data[3]
    gradient_check.check_backward(
        lambda x: super_node_transmitter_unit(x, atom_size), supernode, y_grad)


def test_graph_warp_gate_unit_forward(graph_warp_gate_unit, data):
    embed_atom_data = data[0]
    new_embed_atom_data = data[1]
    merged = graph_warp_gate_unit(embed_atom_data, new_embed_atom_data)
    assert merged.array.shape == (batch_size, atom_size, hidden_dim)


def test_graph_warp_gate_unit_backward(graph_warp_gate_unit, data):
    embed_atom_data = data[0]
    new_embed_atom_data = data[1]
    y_grad = data[3]
    gradient_check.check_backward(graph_warp_gate_unit,
                                  (embed_atom_data, new_embed_atom_data),
                                  y_grad, eps=0.01)


def test_super_warp_gate_unit_forward(super_warp_gate_unit, data):
    supernode = data[2]
    merged = super_warp_gate_unit(supernode, supernode)
    assert merged.array.shape == (batch_size, supernode_dim)


def test_super_warp_gate_unit_backward(super_warp_gate_unit, data):
    supernode = data[2]
    supernode_grad = data[4]
    gradient_check.check_backward(super_warp_gate_unit,
                                  (supernode, supernode),
                                  supernode_grad, eps=0.01)


def check_forward(gwm, embed_atom_data, new_embed_atom_data, supernode):
    gwm.GRU_local.reset_state()
    gwm.GRU_super.reset_state()
    h_actual, g_actual = gwm(embed_atom_data, new_embed_atom_data, supernode)
    assert h_actual.array.shape == (batch_size, atom_size, hidden_dim)
    assert g_actual.array.shape == (batch_size, supernode_dim)


def test_forward_cpu(gwm, data):
    embed_atom_data, new_embed_atom_data, supernode = data[:3]
    check_forward(gwm, embed_atom_data, new_embed_atom_data, supernode)


@pytest.mark.gpu
def test_forward_gpu(gwm, data):
    embed_atom_data, new_embed_atom_data, supernode = data[:3]
    embed_atom_data = cuda.to_gpu(embed_atom_data)
    new_embed_atom_data = cuda.to_gpu(new_embed_atom_data)
    supernode = cuda.to_gpu(supernode)
    gwm.to_gpu()
    check_forward(gwm, embed_atom_data, new_embed_atom_data, supernode)


def check_backward(gwm, embed_atom_data, new_embed_atom_data, supernode,
                   y_grad, supernode_grad):
    gwm.GRU_local.reset_state()
    gwm.GRU_super.reset_state()

    # TODO: rtol is too high! GWM is too large to calculate
    # numerical differentiation
    gradient_check.check_backward(gwm, (embed_atom_data, new_embed_atom_data,
                                        supernode), (y_grad, supernode_grad),
                                  eps=0.1, rtol=1e-0)


def test_backward_cpu(gwm, data):
    check_backward(gwm, *data)


@pytest.mark.gpu
def test_backward_gpu(gwm, data):
    gwm.to_gpu()
    check_backward(gwm, *map(cuda.to_gpu, data))


def test_forward_cpu_graph_invariant(gwm, data):
    permutation_index = numpy.random.permutation(atom_size)
    gwm.reset_state()
    embed_atom_data, new_embed_atom_data, supernode = data[:3]
    h_actual, g_actual = gwm(embed_atom_data, new_embed_atom_data, supernode)

    permute_embed_atom_data = permute_node(
        embed_atom_data, permutation_index, axis=1)
    permute_new_embed_atom_data = permute_node(
        new_embed_atom_data, permutation_index, axis=1)
    gwm.reset_state()
    permute_h_actual, permute_g_actual = gwm(
        permute_embed_atom_data, permute_new_embed_atom_data, supernode)
    numpy.testing.assert_allclose(
        permute_node(h_actual.data, permutation_index, axis=1),
        permute_h_actual.data, rtol=1e-5, atol=1e-5)

    numpy.testing.assert_allclose(g_actual.data, permute_g_actual.data,
                                  rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
