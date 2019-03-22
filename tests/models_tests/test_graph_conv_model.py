import itertools
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.graph_conv_model import GraphConvModel
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links.update.gin_update import GINUpdate
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.readout.gin_readout import GINReadout


atom_size = 5
super_dim = 7
in_channels = 6
out_dim = 4
batch_size = 2
num_edge_type = 3

updates = [GGNNUpdate, GINUpdate]
readouts = [GGNNReadout, GINReadout]
params = list(itertools.product(updates, readouts))


@pytest.fixture(params=params)
def plain_context(request):
    update, readout = request.param
    if update == GGNNUpdate:
        adj_type = 3
    elif update == GINUpdate:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    model = make_model(update, readout)
    return model, data


@pytest.fixture(params=params)
def gwm_context(request):
    update, readout = request.param
    if update == GGNNUpdate:
        adj_type = 3
    elif update == GINUpdate:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    model = make_gwm_model(update, readout)
    return model, data


def make_model(update, readout):
    return GraphConvModel(
        update_layer=update, readout_layer=readout, n_layers=3,
        in_channels=in_channels, n_edge_type=num_edge_type,
        out_dim=out_dim, with_gwm=False)


def make_gwm_model(update, readout):
    return GraphConvModel(
        update_layer=update, readout_layer=readout, n_layers=3,
        in_channels=in_channels, n_edge_type=num_edge_type,
        hidden_dim_super=super_dim, out_dim=out_dim, with_gwm=True)


def make_data(adj_type):
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    if adj_type == 2:
        adj_data = numpy.random.randint(
            0, high=2, size=(batch_size, atom_size, atom_size)
        ).astype(numpy.float32)
    elif adj_type == 3:
        adj_data = numpy.random.randint(
            0, high=2, size=(batch_size, num_edge_type, atom_size, atom_size)
        ).astype(numpy.float32)
    else:
        raise ValueError
    super_data = numpy.random.uniform(-1, 1, (batch_size, super_dim)
                                      ).astype(numpy.float32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)
    return atom_data, adj_data, super_data, y_grad


def test_plain_model_forward(plain_context):
    model, data = plain_context
    atom_array = data[0]
    adj = data[1]
    y_actual = model(atom_array, adj)
    assert y_actual.shape == (batch_size, out_dim)


def test_gwm_model_forward(gwm_context):
    model, data = gwm_context
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim)
