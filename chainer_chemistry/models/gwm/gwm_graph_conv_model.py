import chainer
from chainer import cuda
from chainer import functions
from chainer import links

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.normalization.graph_batch_normalization import GraphBatchNormalization  # NOQA
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.models.gwm.gwm import GWM
from chainer_chemistry.models.relgcn import rescale_adj


def to_array(x):
    """Convert x into numpy.ndarray or cupy.ndarray"""
    if isinstance(x, chainer.Variable):
        x = x.array
    return x


class GWMGraphConvModel(chainer.Chain):
    """Unified module of Graph Convolution Model with GWM

    Note that this module is experimental, all update_layer and
    readout_layer combination is not supported.
    Please refer `test_gwm_graph_conv_model.py` for tested combinations.
    This module might not be maintained in the future.

    Args:
        hidden_channels (int or list): hidden channels for update
        out_dim (int): output dim
        update_layer (chainer.links.Link):
        readout_layer (chainer.links.Link):
        n_update_layers (int or None):
        out_channels (None or lsit):
        super_node_dim (int):
        n_atom_types (int):
        n_edge_types (int):
        dropout_ratio (float):
        with_gwm (bool):
        concat_hidden (bool):
        sum_hidden (bool):
        weight_tying (bool):
        scale_adj (bool):
        activation (callable):
        use_batchnorm (bool):
        n_activation (int or None):
        update_kwargs (dict or None):
        readout_kwargs (dict or None):
        gwm_kwargs (dict or None):
    """
    def __init__(self, hidden_channels, out_dim, update_layer, readout_layer,
                 n_update_layers=None, out_channels=None, super_node_dim=None,
                 n_atom_types=MAX_ATOMIC_NUM, n_edge_types=4,
                 dropout_ratio=-1.0, with_gwm=True,
                 concat_hidden=False, sum_hidden=False, weight_tying=False,
                 scale_adj=False, activation=None, use_batchnorm=False,
                 n_activation=None, update_kwargs=None, readout_kwargs=None,
                 gwm_kwargs=None):
        super(GWMGraphConvModel, self).__init__()

        # General: length of hidden_channels must be n_layers + 1
        if isinstance(hidden_channels, int):
            if n_update_layers is None:
                raise ValueError('n_update_layers is None')
            else:
                hidden_channels = [hidden_channels
                                   for _ in range(n_update_layers + 1)]
        elif isinstance(hidden_channels, list):
            if out_channels is None:
                n_update_layers = len(hidden_channels) - 1
            else:
                n_update_layers = len(hidden_channels)
        else:
            raise TypeError('Unexpected value for hidden_channels {}'
                            .format(hidden_channels))

        if readout_layer == GeneralReadout and hidden_channels[-1] != out_dim:
            # When use GWM, hidden channels must be same. But GeneralReadout
            # cannot change the dimension. So when use General Readout and GWM,
            # hidden channel and out_dim should be same.
            if with_gwm:
                raise ValueError('Unsupported combination.')
            else:
                hidden_channels[-1] = out_dim

        # When use with_gwm, concat_hidden, sum_hidden and weight_tying option,
        # hidden_channels must be same
        if with_gwm or concat_hidden or sum_hidden or weight_tying:
            if not all([in_dim == hidden_channels[0]
                        for in_dim in hidden_channels]):
                raise ValueError(
                    'hidden_channels must be same but different {}'
                    .format(hidden_channels))

        if with_gwm and super_node_dim is None:
            print('[WARNING] super_node_dim is None, set to {}'
                  .format(hidden_channels[0]))
            super_node_dim = hidden_channels[0]

        if out_channels is None:
            in_channels_list = hidden_channels[:-1]
            out_channels_list = hidden_channels[1:]
        else:
            # For RelGAT concat_heads option
            in_channels_list = hidden_channels
            out_channels_list = out_channels
        assert len(in_channels_list) == n_update_layers
        assert len(out_channels_list) == n_update_layers

        n_use_update_layers = 1 if weight_tying else n_update_layers
        n_readout_layers = n_use_update_layers if concat_hidden or sum_hidden else 1  # NOQA
        n_activation = n_use_update_layers if n_activation is None else n_activation  # NOQA

        if update_kwargs is None:
            update_kwargs = {}
        if readout_kwargs is None:
            readout_kwargs = {}
        if gwm_kwargs is None:
            gwm_kwargs = {}

        with self.init_scope():
            self.embed = EmbedAtomID(out_size=hidden_channels[0],
                                     in_size=n_atom_types)
            self.update_layers = chainer.ChainList(
                *[update_layer(in_channels=in_channels_list[i],
                               out_channels=out_channels_list[i],
                               n_edge_types=n_edge_types, **update_kwargs)
                  for i in range(n_use_update_layers)])
            # when use weight_tying option, hidden_channels must be same.
            # So we can use -1 index
            self.readout_layers = chainer.ChainList(
                *[readout_layer(out_dim=out_dim,
                                # in_channels=hidden_channels[-1],
                                in_channels=None,
                                **readout_kwargs)
                  for _ in range(n_readout_layers)])
            if with_gwm:
                self.gwm = GWM(hidden_dim=hidden_channels[0],
                               hidden_dim_super=super_node_dim,
                               n_layers=n_use_update_layers, **gwm_kwargs)
                self.embed_super = links.Linear(None, out_size=super_node_dim)
                self.linear_for_concat_super = links.Linear(in_size=None,
                                                            out_size=out_dim)
            if use_batchnorm:
                self.bnorms = chainer.ChainList(
                    *[GraphBatchNormalization(
                        out_channels_list[i]) for i in
                        range(n_use_update_layers)])

        self.readout_layer = readout_layer
        self.update_layer = update_layer
        self.weight_tying = weight_tying
        self.with_gwm = with_gwm
        self.concat_hidden = concat_hidden
        self.sum_hidden = sum_hidden
        self.scale_adj = scale_adj
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm
        self.n_activation = n_activation
        self.n_update_layers = n_update_layers
        self.n_edge_types = n_edge_types

    def __call__(self, atom_array, adj, super_node=None, is_real_node=None):
        self.reset_state()

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)
        else:
            # TODO(nakago): GraphLinear or GraphMLP can be used.
            h = atom_array

        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        if self.with_gwm:
            h_s = self.embed_super(super_node)

        additional_kwargs = self.preprocess_addtional_kwargs(
            atom_array, adj, super_node=super_node, is_real_node=is_real_node)

        if self.scale_adj:
            adj = rescale_adj(adj)

        g_list = []
        for step in range(self.n_update_layers):
            update_layer_index = 0 if self.weight_tying else step
            h_new = self.update_layers[update_layer_index](
                h=h, adj=adj, **additional_kwargs)

            if self.with_gwm:
                h_new, h_s = self.gwm(h, h_new, h_s, update_layer_index)
            h = h_new

            if self.use_batchnorm:
                h = self.bnorms[update_layer_index](h)

            if self.dropout_ratio > 0.:
                h = functions.dropout(h, ratio=self.dropout_ratio)

            if self.activation is not None and step < self.n_activation:
                h = self.activation(h)

            if self.concat_hidden or self.sum_hidden:
                g = self.readout_layers[step](
                    h=h, h0=h0, is_real_node=is_real_node, **additional_kwargs)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            if self.sum_hidden:
                g = functions.sum(functions.stack(g_list), axis=0)
            else:
                g = self.readout_layers[0](
                    h=h, h0=h0, is_real_node=is_real_node)
            if self.with_gwm:
                g = functions.concat((g, h_s), axis=1)
                g = functions.relu(self.linear_for_concat_super(g))
            return g

    def reset_state(self):
        if hasattr(self.update_layers[0], 'reset_state'):
            [update_layer.reset_state() for update_layer in self.update_layers]

        if self.with_gwm:
            self.gwm.reset_state()

    def preprocess_addtional_kwargs(self, *args, **kwargs):
        return {}
