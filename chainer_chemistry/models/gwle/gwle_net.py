from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import GINUpdate, NFPReadout, NFPUpdate, \
    RSGCNUpdate, GeneralReadout  # NOQA
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links.update.relgat_update import RelGATUpdate
from chainer_chemistry.links.update.relgcn_update \
    import RelGCNUpdate, RelGCNSparseUpdate


from chainer_chemistry.models.gwle.gwle_graph_conv_model import GWLEGraphConvModel  # NOQA
from chainer_chemistry.models.cwle.cwle_graph_conv_model import to_array
from chainer_chemistry.models.cwle.cwle_graph_conv_model import MAX_WLE_NUM


class GGNN_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_wle=True, n_wle_types=MAX_WLE_NUM):
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(GGNN_GWLE, self).__init__(
            update_layer=GGNNUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            weight_tying=weight_tying, n_edge_types=n_edge_types,
            with_wle=with_wle, readout_kwargs=readout_kwargs,
            n_wle_types=n_wle_types)


class RelGCN_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_wle=True, n_wle_types=MAX_WLE_NUM):
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(RelGCN_GWLE, self).__init__(
            update_layer=RelGCNUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            weight_tying=weight_tying, n_edge_types=n_edge_types,
            with_wle=with_wle, readout_kwargs=readout_kwargs,
            n_wle_types=n_wle_types)


class RelGAT_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_wle=True, n_wle_types=MAX_WLE_NUM):
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(RelGAT_GWLE, self).__init__(
            update_layer=RelGATUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            weight_tying=weight_tying, n_edge_types=n_edge_types,
            with_wle=with_wle, readout_kwargs=readout_kwargs,
            n_wle_types=n_wle_types)


class GIN_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=16,
                 n_update_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_wle=True, n_wle_types=MAX_WLE_NUM):
        update_kwargs = {'dropout_ratio': dropout_ratio}
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(GIN_GWLE, self).__init__(
            update_layer=GINUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers, n_atom_types=n_atom_types,
            concat_hidden=concat_hidden, weight_tying=weight_tying,
            n_edge_types=n_edge_types, with_wle=with_wle,
            update_kwargs=update_kwargs, readout_kwargs=readout_kwargs,
            n_wle_types=n_wle_types)


class NFP_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 max_degree=6, n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False, with_wle=True, n_wle_types=MAX_WLE_NUM):
        update_kwargs = {'max_degree': max_degree}
        super(NFP_GWLE, self).__init__(
            update_layer=NFPUpdate, readout_layer=NFPReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            sum_hidden=True, with_wle=with_wle, update_kwargs=update_kwargs,
            n_wle_types=n_wle_types)
        self.max_degree = max_degree
        self.n_degree_type = max_degree + 1
        self.ch0 = hidden_channels

    def preprocess_addtional_kwargs(self, *args, **kwargs):
        atom_array, adj = args[:2]
        bs, num_node = atom_array.shape[:2]
        # For NFP Update
        if adj.ndim == 4:
            degree_mat = self.xp.sum(to_array(adj), axis=(1, 2))
        elif adj.ndim == 3:
            degree_mat = self.xp.sum(to_array(adj), axis=1)
        else:
            raise ValueError('Unexpected value adj '
                             .format(adj.shape))
        # deg_conds: (minibatch, atom, ch)
        deg_conds = [self.xp.broadcast_to(
            ((degree_mat - degree) == 0)[:, :, None],
            (bs, num_node, self.ch0))
            for degree in range(1, self.n_degree_type + 1)]
        return {'deg_conds': deg_conds}


class RSGCN_GWLE(GWLEGraphConvModel):
    def __init__(self, out_dim, hidden_channels=32, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None, dropout_ratio=0.5,
                 with_wle=True, n_wle_types=MAX_WLE_NUM):
        if readout is None:
            readout = GeneralReadout
        super(RSGCN_GWLE, self).__init__(
            update_layer=RSGCNUpdate, readout_layer=readout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers, n_atom_types=n_atom_types,
            use_batchnorm=use_batch_norm, activation=functions.relu,
            n_activation=n_update_layers-1, dropout_ratio=dropout_ratio,
            with_wle=with_wle, n_wle_types=n_wle_types)
