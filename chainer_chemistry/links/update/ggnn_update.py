import chainer
from chainer import functions
from chainer import links

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GGNNUpdate(chainer.Chain):
    """GGNN submodule for update part.

    Args:
        hidden_dim (int): dimension of feature vector associated to
            each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        num_edge_type (int): number of types of edge
        weight_tying (bool): enable weight_tying or not
    """

    def __init__(self, hidden_dim=16, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, num_edge_type=4,
                 weight_tying=True):
        super(GGNNUpdate, self).__init__()
        n_layer = 1 if weight_tying else n_layers
        with self.init_scope():
            self.graph_linears = chainer.ChainList(
                *[GraphLinear(hidden_dim, num_edge_type * hidden_dim)
                  for _ in range(n_layer)])
            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
        self.n_layers = n_layers
        self.num_edge_type = num_edge_type
        self.weight_tying = weight_tying

    def __call__(self, h, adj, step=0):
        # --- Message part ---
        mb, atom, ch = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        m = functions.reshape(self.graph_linears[message_layer_index](h),
                              (mb, atom, out_ch, self.num_edge_type))
        # m: (minibatch, atom, ch, edge_type)
        # Transpose
        m = functions.transpose(m, (0, 3, 1, 2))
        # m: (minibatch, edge_type, atom, ch)

        adj = functions.reshape(adj, (mb * self.num_edge_type, atom, atom))
        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb * self.num_edge_type, atom, out_ch))

        m = chainer_chemistry.functions.matmul(adj, m)

        # (minibatch * edge_type, atom, out_ch)
        m = functions.reshape(m, (mb, self.num_edge_type, atom, out_ch))
        m = functions.sum(m, axis=1)
        # (minibatch, atom, out_ch)

        # --- Update part ---
        # Contraction
        h = functions.reshape(h, (mb * atom, ch))

        # Contraction
        m = functions.reshape(m, (mb * atom, ch))

        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion
        out_h = functions.reshape(out_h, (mb, atom, ch))
        return out_h

    def reset_state(self):
        self.update_layer.reset_state()
