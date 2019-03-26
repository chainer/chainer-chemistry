import chainer
import chainer_chemistry

from chainer import cuda
from chainer import links
from chainer import functions
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.gwm import GWM


def to_array(x):
    """Convert x into numpy.ndarray or cupy.ndarray"""
    if isinstance(x, chainer.Variable):
        x = x.array
    return x


def rescale_adj(adj):
    """Normalize adjacency matrix
    It ensures that activations are on a similar scale irrespective of
    the number of neighbors

    Args:
        adj (:class:`chainer.Variable`, or :class:`numpy.ndarray` \
        or :class:`cupy.ndarray`):
            adjacency matrix

    Returns:
        :class:`chainer.Variable`: normalized adjacency matrix

    """
    xp = cuda.get_array_module(adj)
    num_neighbors = functions.sum(adj, axis=(1, 2))
    base = xp.ones(num_neighbors.shape, dtype=xp.float32)
    cond = num_neighbors.data != 0
    num_neighbors_inv = 1 / functions.where(cond, num_neighbors, base)
    return adj * functions.broadcast_to(
        num_neighbors_inv[:, None, None, :], adj.shape)


class GraphConvModel(chainer.Chain):
    def __init__(self, hidden_channels, out_dim, update_layer, readout_layer, n_layers=None,
                 out_channels=None, hidden_dim_super=None, n_atom_types=MAX_ATOMIC_NUM,
                 n_edge_types=4, max_degree=6, dropout_ratio=-1.0, with_gwm=True,
                 concat_hidden=False, sum_hidden=False, weight_tying=False,
                 scale_adj=False, activation=None, use_batchnorm=False, n_activation=None,
                 update_kwargs=None, readout_kwargs=None, gwm_kwargs=None):
        # Note: in_channels can be integer or list
        # Note: Is out_dim necessary?
        super(GraphConvModel, self).__init__()

        # General: length of hidden_channels must be n_layers + 1

        # When use with_gwm, concat_hidden, sum_hidden option,
        # hidden_channels must be same
        if with_gwm or concat_hidden or sum_hidden:
            if isinstance(hidden_channels, list):
                if all([in_dim == hidden_channels[0] for in_dim in hidden_channels]):
                    raise ValueError
            if with_gwm and hidden_dim_super is None:
                raise ValueError

        if isinstance(hidden_channels, int):
            if n_layers is None:
                raise ValueError
            hidden_channels = [hidden_channels for _ in range(n_layers + 1)]
        elif isinstance(hidden_channels, list):
            n_layers = len(hidden_channels) - 1
        else:
            raise TypeError

        if readout_layer == GeneralReadout and hidden_channels[-1] != out_dim:
            hidden_channels[-1] = out_dim
            # When use GWM, hidden channels must be same. But GeneralReadout
            # cannot change the dimension. So before apply readout, it is
            # necessary to change the dimension to out_dim.
            # TODO: raise warning
            if with_gwm:
                n_layers -= 1

        if out_channels is None:
            in_channels_list = hidden_channels[:-1]
            out_channels_list = hidden_channels[1:]
        else:
            # For RelGAT concat_heads option
            in_channels_list = hidden_channels
            out_channels_list = out_channels
        assert len(in_channels_list) >= n_layers
        assert len(out_channels_list) >= n_layers

        n_update_layers = 1 if weight_tying else n_layers
        n_readout_layers = n_layers if concat_hidden or sum_hidden else 1
        n_degree_type = max_degree + 1
        n_activation = n_layers if n_activation is None else n_activation

        if update_kwargs is None:
            update_kwargs = {}
        if readout_kwargs is None:
            readout_kwargs = {}
        if gwm_kwargs is None:
            gwm_kwargs = {}

        with self.init_scope():
            self.embed = EmbedAtomID(out_size=hidden_channels[0], in_size=n_atom_types)
            self.update_layers = chainer.ChainList(
                *[update_layer(in_channels=in_channels_list[i], out_channels=out_channels_list[i],
                               n_edge_types=n_edge_types, **update_kwargs)
                  for i in range(n_update_layers)])
            self.readout_layers = chainer.ChainList(
                *[readout_layer(out_dim=out_dim, in_channels=hidden_channels[-1], **readout_kwargs)
                  for _ in range(n_readout_layers)])
            if with_gwm:
                self.gwm = GWM(hidden_dim=hidden_channels[0], hidden_dim_super=hidden_dim_super,
                               n_layers=n_update_layers, **gwm_kwargs)
                self.embed_super = links.Linear(None, out_size=hidden_dim_super)
                self.linear_for_concat_super = links.Linear(in_size=None, out_size=out_dim)
            if use_batchnorm:
                # TODO: check
                self.bnorms = chainer.ChainList(
                    *[chainer_chemistry.links.GraphBatchNormalization(
                        out_channels[i]) for i in range(n_update_layers)]
                )

        self.readout_layer = readout_layer
        self.update_layer = update_layer
        self.n_layers = n_layers
        self.weight_tying = weight_tying
        self.with_gwm = with_gwm
        self.concat_hidden = concat_hidden
        self.sum_hidden = sum_hidden
        self.n_degree_type = n_degree_type
        self.scale_adj = scale_adj
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm
        self.n_activation = n_activation

    def __call__(self, atom_array, adj, super_node=None, is_real_node=None):
        self.reset_state()

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)
        else:
            # TODO: GraphLinear or GraphMLP can be used.
            # TODO: RelGCN use GraphLinear here.
            h = atom_array

        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        if self.with_gwm:
            h_s = self.embed_super(super_node)

        # For NFP Update
        if adj.ndim == 4:
            degree_mat = self.xp.sum(to_array(adj), axis=(1, 2))
        elif adj.ndim == 3:
            degree_mat = self.xp.sum(to_array(adj), axis=1)
        else:
            raise ValueError
        # deg_conds: (minibatch, atom, ch)
        deg_conds = [self.xp.broadcast_to(
            ((degree_mat - degree) == 0)[:, :, None], h.shape)
            for degree in range(1, self.n_degree_type + 1)]

        if self.scale_adj:
            adj = rescale_adj(adj)

        g_list = []
        for step in range(self.n_layers):
            update_layer_index = 0 if self.weight_tying else step
            h_new = self.update_layers[update_layer_index](h=h, adj=adj, deg_conds=deg_conds)

            if self.with_gwm:
                h_new, h_s = self.gwm(h, h_new, h_s, update_layer_index)
            h = h_new

            if self.use_batchnorm:
                h = self.bnorms[update_layer_index]

            if self.dropout_ratio > 0.:
                h = functions.dropout(h, ratio=self.dropout_ratio)

            if self.activation is not None and step < self.n_activation:
                h = self.activation(h)

            if self.concat_hidden or self.sum_hidden:
                g = self.readout_layers[step](
                    h=h, h0=h0, is_real_node=is_real_node)
                g_list.append(g)

        # TODO: need cool implementation
        if self.readout_layer == GeneralReadout and self.with_gwm:
            h = self.update_layers[-1](h=h, adj=adj, deg_conds=deg_conds)

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