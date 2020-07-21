import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GNNFiLMUpdate(chainer.Chain):
    """GNNFiLM submodule for update part.

    Args:
        hidden_channels (int): dimension of feature vector associated to
            each atom
        n_edge_types (int): number of types of edge
    """

    def __init__(self, hidden_channels=16, n_edge_types=5,
                 activation=functions.relu):
        super(GNNFiLMUpdate, self).__init__()
        self.n_edge_types = n_edge_types
        self.activation = activation
        with self.init_scope():
            self.W_linear = GraphLinear(
                in_size=None, out_size=self.n_edge_types * hidden_channels,
                nobias=True)  # W_l in eq. (6)
            self.W_g = GraphLinear(
                in_size=None, out_size=self.n_edge_types * hidden_channels * 2,
                nobias=True)  # g in eq. (6)
            self.norm_layer = links.LayerNormalization()  # l in eq. (6)

    def forward(self, h, adj):
        # --- Message part ---

        xp = self.xp
        mb, atom, ch = h.shape
        newshape = adj.shape + (ch, )
        adj = functions.broadcast_to(adj[:, :, :, :, xp.newaxis], newshape)
        messages = functions.reshape(self.W_linear(h),
                                     (mb, atom, ch, self.n_edge_types))
        messages = functions.transpose(messages, (3, 0, 1, 2))
        film_weights = functions.reshape(self.W_g(h),
                                         (mb, atom, 2 * ch, self.n_edge_types))
        film_weights = functions.transpose(film_weights, (3, 0, 1, 2))
        # (n_edge_types, minibatch, atom, out_ch)
        gamma = film_weights[:, :, :, :ch]
        # (n_edge_types, minibatch, atom, out_ch)
        beta = film_weights[:, :, :, ch:]

        # --- Update part ---

        messages = functions.expand_dims(
            gamma, axis=3) * functions.expand_dims(
            messages, axis=2) + functions.expand_dims(beta, axis=3)
        messages = self.activation(messages)
        # (minibatch, n_edge_types, atom, atom, out_ch)
        messages = functions.transpose(messages, (1, 0, 2, 3, 4))
        messages = adj * messages
        messages = functions.sum(messages, axis=3)  # sum across atoms
        messages = functions.sum(messages, axis=1)  # sum across n_edge_types
        messages = functions.reshape(messages, (mb * atom, ch))
        messages = self.norm_layer(messages)
        messages = functions.reshape(messages, (mb, atom, ch))
        return messages
