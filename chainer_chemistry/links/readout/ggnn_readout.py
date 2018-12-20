import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GGNNReadout(chainer.Chain):
    """GGNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each atom
        nobias (bool): If ``True``, then this function does not use
            the bias
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function
            It can be replaced with the functions.identity.
    """

    def __init__(self, out_dim, hidden_dim=16, nobias=False,
                 activation=functions.tanh):
        super(GGNNReadout, self).__init__()
        with self.init_scope():
            self.i_layer = GraphLinear(None, out_dim, nobias=nobias)
            self.j_layer = GraphLinear(None, out_dim, nobias=nobias)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.nobias = nobias
        self.activation = activation

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
        g = self.activation(functions.sum(g, axis=1))
        return g
