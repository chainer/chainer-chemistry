import itertools
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.readout.nfp_readout import NFPReadout
from chainer_chemistry.links.readout.schnet_readout import SchNetReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links.update.gin_update import GINUpdate
from chainer_chemistry.links.update.relgat_update import RelGATUpdate
from chainer_chemistry.links.update.relgcn_update import RelGCNUpdate
from chainer_chemistry.links.update.rsgcn_update import RSGCNUpdate
from chainer_chemistry.models.gwm.gwm_graph_conv_model import GWMGraphConvModel


atom_size = 5
super_dim = 7
in_channels = 6
out_dim = 4
batch_size = 2
n_edge_types = 3

# TODO(nakago): SchNetUpdate need `in_channels` kwargs, not supported.
updates_2dim = [GINUpdate, RSGCNUpdate]
# TODO(nakago): Support MPNNUpdate.
updates_3dim = [GGNNUpdate, RelGATUpdate, RelGCNUpdate]
updates = updates_2dim + updates_3dim

# TODO(nakago): MPNNReadout need to specify `in_channels` and not supported.
readouts = [GGNNReadout, NFPReadout, SchNetReadout]
hidden_channels = [[6, 6, 6, 6], 6]
use_bn = [True, False]
use_weight_tying = [True, False]

params = list(itertools.product(
    updates, readouts, hidden_channels, use_bn, use_weight_tying,
))


@pytest.fixture(params=params)
def plain_context(request):
    update, readout, ch, bn, wt = request.param
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    model = make_model(update, readout, ch, bn, wt)
    return model, data


@pytest.fixture(params=params)
def gwm_context(request):
    update, readout, ch, bn, wt = request.param
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    model = make_gwm_model(update, readout, ch, bn, wt)
    return model, data


def make_model(update, readout, ch, bn, wt):
    # print('update', update, 'readout', readout, 'ch', ch, 'bn', bn, 'wt', wt)
    return GWMGraphConvModel(
        update_layer=update, readout_layer=readout, n_update_layers=3,
        hidden_channels=ch, n_edge_types=n_edge_types, weight_tying=wt,
        out_dim=out_dim, with_gwm=False, use_batchnorm=bn)


def make_gwm_model(update, readout, ch, bn, wt):
    return GWMGraphConvModel(
        update_layer=update, readout_layer=readout, n_update_layers=3,
        hidden_channels=ch, n_edge_types=n_edge_types, weight_tying=wt,
        super_node_dim=super_dim, out_dim=out_dim, with_gwm=True,
        use_batchnorm=bn)


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
            0, high=2, size=(batch_size, n_edge_types, atom_size, atom_size)
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
    if model.weight_tying:
        assert len(model.update_layers) == 1
    else:
        assert len(model.update_layers) == 3


def test_gwm_model_forward(gwm_context):
    model, data = gwm_context
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim)
    if model.weight_tying:
        assert len(model.update_layers) == 1
    else:
        assert len(model.update_layers) == 3


# SchNet is not supported
sp_params = list(itertools.product(
    updates_2dim[:-1] + updates_3dim,
    [[6, 6, 6, 6], [4, 4, 4, 4], [6, 5, 3, 4]],
))


@pytest.mark.parametrize(('update', 'ch'), sp_params)
def test_plain_model_forward_general_readout(
        update, ch):
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    model = GWMGraphConvModel(update_layer=update,
                              readout_layer=GeneralReadout,
                              hidden_channels=ch,
                              out_dim=out_dim,
                              n_edge_types=n_edge_types,
                              with_gwm=False)
    atom_array = data[0]
    adj = data[1]
    y_actual = model(atom_array, adj)
    assert y_actual.shape == (batch_size, out_dim)


@pytest.mark.parametrize('update',
                         updates_2dim[:-1] + updates_3dim)
def test_gwm_model_forward_general_readout(update):
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    ch = [6, 6, 6, 6]
    with pytest.raises(ValueError):
        model = GWMGraphConvModel(update_layer=update,
                                  readout_layer=GeneralReadout,
                                  hidden_channels=ch,
                                  out_dim=out_dim,
                                  n_edge_types=n_edge_types,
                                  super_node_dim=super_dim,
                                  with_gwm=True)
    ch = [4, 4, 4, 4]
    model = GWMGraphConvModel(update_layer=update,
                              readout_layer=GeneralReadout,
                              hidden_channels=ch,
                              out_dim=out_dim,
                              n_edge_types=n_edge_types,
                              super_node_dim=super_dim,
                              with_gwm=True)
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim)


p = list(itertools.product(updates_2dim[:-1] + updates_3dim, readouts,
                           [True, False]))


@pytest.mark.parametrize(('update', 'readout', 'gwm'), p)
def test_model_forward_general_weight_tying(update, readout, gwm):
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    ch = [6, 7, 8, 6]
    if gwm:
        with pytest.raises(ValueError):
            model = GWMGraphConvModel(update_layer=update,
                                      readout_layer=GeneralReadout,
                                      hidden_channels=ch,
                                      out_dim=out_dim,
                                      n_edge_types=n_edge_types,
                                      super_node_dim=super_dim,
                                      with_gwm=gwm)
    else:
        model = GWMGraphConvModel(update_layer=update,
                                  readout_layer=GeneralReadout,
                                  hidden_channels=ch,
                                  out_dim=out_dim,
                                  n_edge_types=n_edge_types,
                                  super_node_dim=super_dim,
                                  with_gwm=gwm)
        atom_array = data[0]
        adj = data[1]
        super_node = data[2]  # NOQA
        y_actual = model(atom_array, adj)
        assert y_actual.shape == (batch_size, out_dim)


@pytest.mark.parametrize(('update', 'readout', 'gwm'), p)
def test_model_forward_general_concat_hidden(update, readout, gwm):
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    ch = [6, 6, 6, 6]
    model = GWMGraphConvModel(update_layer=update,
                              readout_layer=readout,
                              hidden_channels=ch,
                              out_dim=out_dim,
                              n_edge_types=n_edge_types,
                              super_node_dim=super_dim,
                              concat_hidden=True,
                              with_gwm=gwm)
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim * (len(ch) - 1))


@pytest.mark.parametrize(('update', 'readout', 'gwm'), p)
def test_model_forward_general_sum_hidden(update, readout, gwm):
    if update in updates_3dim:
        adj_type = 3
    elif update in updates_2dim:
        adj_type = 2
    else:
        raise ValueError
    data = make_data(adj_type)
    ch = [6, 6, 6, 6]
    model = GWMGraphConvModel(update_layer=update,
                              readout_layer=readout,
                              hidden_channels=ch,
                              out_dim=out_dim,
                              n_edge_types=n_edge_types,
                              super_node_dim=super_dim,
                              sum_hidden=True,
                              with_gwm=gwm)
    atom_array = data[0]
    adj = data[1]
    super_node = data[2]
    y_actual = model(atom_array, adj, super_node)
    assert y_actual.shape == (batch_size, out_dim)


if __name__ == '__main__':
    # -x is to stop when first failed.
    pytest.main([__file__, '-v', '-s', '-x'])
