import chainer
from chainer import functions
from chainer import links

import chainer_chemistry
from chainer_chemistry.links.connection.graph_linear import GraphLinear
from chainer_chemistry.utils import is_sparse


class GGNNUpdate(chainer.Chain):
    """GGNN submodule for update part.

    Args:
        in_channels (int or None): input dim of feature vector for each node
        hidden_channels (int): dimension of feature vector for each node
        out_channels (int or None): output dime of feature vector for each node
            When `None`, `hidden_channels` is used.
        n_edge_types (int): number of types of edge
    """

    def __init__(self, in_channels=None, hidden_channels=16,
                 out_channels=None, n_edge_types=4, **kwargs):
        if out_channels is None:
            out_channels = hidden_channels
        super(GGNNUpdate, self).__init__()
        if in_channels is None:
            gru_in_channels = None
        else:
            gru_in_channels = in_channels + hidden_channels
        with self.init_scope():
            self.graph_linear = GraphLinear(
                in_channels, n_edge_types * hidden_channels)
            self.update_layer = links.GRU(gru_in_channels, out_channels)
        self.n_edge_types = n_edge_types
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def __call__(self, h, adj, **kwargs):
        hidden_ch = self.hidden_channels
        # --- Message part ---
        mb, atom, in_ch = h.shape
        m = functions.reshape(self.graph_linear(h),
                              (mb, atom, hidden_ch, self.n_edge_types))
        # m: (minibatch, atom, ch, edge_type)
        # Transpose
        m = functions.transpose(m, (0, 3, 1, 2))
        # m: (minibatch, edge_type, atom, ch)

        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb * self.n_edge_types, atom, hidden_ch))

        if is_sparse(adj):
            m = functions.sparse_matmul(adj, m)
        else:
            adj = functions.reshape(adj, (mb * self.n_edge_types, atom, atom))
            m = chainer_chemistry.functions.matmul(adj, m)

        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb, self.n_edge_types, atom, hidden_ch))
        m = functions.sum(m, axis=1)
        # (minibatch, atom, out_ch)

        # --- Update part ---
        # Contraction
        h = functions.reshape(h, (mb * atom, in_ch))

        # Contraction
        m = functions.reshape(m, (mb * atom, hidden_ch))

        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion
        out_h = functions.reshape(out_h, (mb, atom, self.out_channels))
        return out_h

    def reset_state(self):
        self.update_layer.reset_state()
