import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GGNNReadout(chainer.Chain):
    """GGNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        in_channels (int or None): dimension of feature vector associated to
            each node. `in_channels` is the total dimension of `h` and `h0`.
        nobias (bool): If ``True``, then this function does not use
            the bias
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function for node representation
            `functions.tanh` was suggested in original paper.
        activation_agg (~chainer.Function or ~chainer.FunctionNode):
            activate function for aggregation
            `functions.tanh` was suggested in original paper.
    """

    def __init__(self, out_dim, in_channels=None, nobias=False,
                 activation=functions.identity,
                 activation_agg=functions.identity):
        super(GGNNReadout, self).__init__()
        with self.init_scope():
            self.i_layer = GraphLinear(in_channels, out_dim, nobias=nobias)
            self.j_layer = GraphLinear(in_channels, out_dim, nobias=nobias)
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.nobias = nobias
        self.activation = activation
        self.activation_agg = activation_agg

    def __call__(self, h, h0=None, is_real_node=None):
        # --- Readout part ---
        # h, h0: (minibatch, node, ch)
        # is_real_node: (minibatch, node)
        h1 = functions.concat((h, h0), axis=2) if h0 is not None else h

        g1 = functions.sigmoid(self.i_layer(h1))
        g2 = self.activation(self.j_layer(h1))
        g = g1 * g2
        if is_real_node is not None:
            # mask virtual node feature to be 0
            mask = self.xp.broadcast_to(
                is_real_node[:, :, None], g.shape)
            g = g * mask
        # sum along node axis
        g = self.activation_agg(functions.sum(g, axis=1))
        return g
