import chainer
from chainer import functions

from chainer_chemistry.links import GraphLinear


class GGNNReadout(chainer.Chain):
    """GGNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each atom
        nobias (bool): If ``True``, then this function does not use
            the bias
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function for node representation
            `functions.tanh` was suggested in original paper.
        activation_agg (~chainer.Function or ~chainer.FunctionNode):
            activate function for aggregation
            `functions.tanh` was suggested in original paper.
    """

    def __init__(self, out_dim, hidden_dim=16, nobias=False,
                 activation=functions.identity,
                 activation_agg=functions.identity):
        super(GGNNReadout, self).__init__()
        with self.init_scope():
            self.i_layer = GraphLinear(None, out_dim, nobias=nobias)
            self.j_layer = GraphLinear(None, out_dim, nobias=nobias)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.nobias = nobias
        self.activation = activation
        self.activation_agg = activation_agg

    def __call__(self, h, h0=None):
        # --- Readout part ---
        # h, h0: (minibatch, atom, ch)
        h1 = functions.concat((h, h0), axis=2) if h0 is not None else h

        g1 = functions.sigmoid(self.i_layer(h1))
        g2 = self.activation(self.j_layer(h1))
        # sum along atom's axis
        g = self.activation_agg(functions.sum(g1 * g2, axis=1))
        return g
