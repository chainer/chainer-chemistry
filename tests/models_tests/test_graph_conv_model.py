import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.graph_conv_model import GraphConvModel
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout


atom_size = 5
super_dim = 7
hidden_dim = 6
out_dim = 4
batch_size = 2
num_edge_type = 3


@pytest.fixture
def plain_model():
    return GraphConvModel(
        update_layer=GGNNUpdate, readout_layer=GGNNReadout, n_layers=3,
        hidden_dim=hidden_dim, n_edge_type=num_edge_type,
        out_dim=out_dim, with_gwm=False)


@pytest.fixture
def gwm_model():
    return GraphConvModel(
        update_layer=GGNNUpdate, readout_layer=GGNNReadout, n_layers=3,
        hidden_dim=hidden_dim, n_edge_type=num_edge_type,
        hidden_dim_super=super_dim, out_dim=out_dim, with_gwm=True)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size, atom_size)
    ).astype(numpy.float32)
    super_data = numpy.random.uniform(-1, 1, (batch_size, super_dim)
                                      ).astype(numpy.float32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)
    return atom_data, adj_data, super_data, y_grad


def test_plain_model_forward(plain_model, data):
    atom_array = data[0]
    adj = data[1]
    y_actual = plain_model(atom_array, adj)
    assert y_actual.shape == (batch_size, out_dim)


def test_gwm_model_forward(gwm_model, data):
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = gwm_model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim)
