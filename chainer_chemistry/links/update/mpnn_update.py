# from typing import Optional  # NOQA

import chainer
from chainer import functions
from chainer import links

import chainer_chemistry


class MPNNUpdate(chainer.Chain):
    r"""MPNN submodule for update part.

    See: Justin Gilmer+, \
        Neural Message Passing for Quantum Chemistry. April 2017.
        `arXiv:1704.01212 <https://arxiv.org/abs/1704.01212>`

    Args:
        hidden_dim (int): dimension of feature vector associated to
            each atom
        nn (~chainer.Link):

    """

    def __init__(self, hidden_dim=16, nn=None):
        # type: (int, Optional[chainer.Link]) -> None
        super(MPNNUpdate, self).__init__()
        with self.init_scope():
            self.message_layer = EdgeNet(out_channels=hidden_dim, nn=nn)
            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.nn = nn

    def __call__(self, h, adj):
        # type: (chainer.Variable, chainer.Variable) -> chainer.Variable
        # adj: (mb, edge_type, node, node)
        mb, node, ch = h.shape
        h = self.message_layer(h, adj)  # h: (mb, node, hidden_dim*2)
        h = functions.reshape(h, (mb * node, self.hidden_dim * 2))
        h = self.update_layer(h)  # h: (mb*node, hidden_dim)
        h = functions.reshape(h, (mb, node, self.hidden_dim))
        return h

    def reset_state(self):
        self.update_layer.reset_state()


class EdgeNet(chainer.Chain):
    """MPNN submodule for message part.

    Edge Network expands edge vector dimension to (d x d) matrix.
    If undirected graph, adj_in and adj_out are same.

    Args:
        out_channels (int): dimension of output feature vector
        nn (~chainer.Link):

    """

    def __init__(self, out_channels, nn=None):
        # type: (int, chainer.Link) -> None
        super(EdgeNet, self).__init__()
        if nn is None:
            nn = chainer.Sequential(
                links.Linear(None, 16), functions.relu,
                links.Linear(None, out_channels**2))
        with self.init_scope():
            if isinstance(nn, chainer.Link):
                self.nn_layer_in = nn
                self.nn_layer_out = nn
        if not isinstance(nn, chainer.Link):
            self.nn_layer_in = nn
            self.nn_layer_out = nn
        self.out_channels = out_channels

    def __call__(self, h, adj):
        # type: (chainer.Variable, chainer.Variable) -> chainer.Variable
        mb, node, ch = h.shape
        if ch != self.out_channels:
            raise ValueError('out_channels must be equal to dimension '
                             'of feature vector associated to each atom, '
                             '{}, but it was set to {}'.format(
                                 ch, self.out_channels))
        # adj: (mb, edge_type, node, node)
        edge_type = adj.shape[1]
        adj_in = adj
        adj_out = functions.transpose(adj, axes=(0, 1, 3, 2))

        # expand edge vector to matrix
        adj_in = functions.reshape(adj_in, (-1, edge_type))
        # adj_in: (mb*node*node, edge_type)
        adj_in = self.nn_layer_in(adj_in)
        # adj_in: (mb*node*node, out_ch*out_ch)
        adj_in = functions.reshape(adj_in, (mb, node, node, ch, ch))
        adj_in = functions.reshape(
            functions.transpose(adj_in, axes=(0, 1, 3, 2, 4)),
            (mb, node * ch, node * ch))

        adj_out = functions.reshape(adj_out, (-1, edge_type))
        # adj_out: (mb*node*node, edge_type)
        adj_out = self.nn_layer_out(adj_out)
        # adj_out: (mb*node*node, out_ch*out_ch)
        adj_out = functions.reshape(adj_out, (mb, node, node, ch, ch))
        adj_out = functions.reshape(
            functions.transpose(adj_out, axes=(0, 1, 3, 2, 4)),
            (mb, node * ch, node * ch))

        # calculate message
        h = functions.reshape(h, (mb, node * ch, 1))
        message_in = chainer_chemistry.functions.matmul(adj_in, h)
        # message_in: (mb, node*ch, 1)
        message_in = functions.reshape(message_in, (mb, node, ch))
        # message_in: (mb, node, out_ch)
        message_out = chainer_chemistry.functions.matmul(adj_out, h)
        # message_out: (mb, node*ch, 1)
        message_out = functions.reshape(message_out, (mb, node, ch))
        message = functions.concat([message_in, message_out], axis=2)
        return message  # message: (mb, node, out_ch * 2)
