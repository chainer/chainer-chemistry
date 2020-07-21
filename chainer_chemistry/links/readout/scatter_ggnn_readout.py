import numpy

import chainer
from chainer import functions


class ScatterGGNNReadout(chainer.Chain):
    """GGNN submodule for readout part using scatter operation.

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
        concat_n_info (bool): If ``True``, node information is concated
            to the result.
    """

    def __init__(self, out_dim, in_channels=None, nobias=False,
                 activation=functions.identity,
                 activation_agg=functions.identity,
                 concat_n_info=False):
        super(ScatterGGNNReadout, self).__init__()
        self.concat_n_info = concat_n_info
        if self.concat_n_info:
            out_dim -= 1
        with self.init_scope():
            self.i_layer = chainer.links.Linear(
                in_channels, out_dim, nobias=nobias)
            self.j_layer = chainer.links.Linear(
                in_channels, out_dim, nobias=nobias)
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.nobias = nobias
        self.activation = activation
        self.activation_agg = activation_agg

    def __call__(self, h, batch, h0=None, is_real_node=None):
        # --- Readout part ---
        h1 = functions.concat((h, h0), axis=1) if h0 is not None else h

        g1 = functions.sigmoid(self.i_layer(h1))
        g2 = self.activation(self.j_layer(h1))
        g = g1 * g2

        # sum along node axis
        y = self.xp.zeros((int(batch[-1]) + 1, self.out_dim),
                          dtype=numpy.float32)
        y = functions.scatter_add(y, batch, g)
        y = self.activation_agg(y)

        if self.concat_n_info:
            n_nodes = self.xp.zeros(y.shape[0], dtype=self.xp.float32)
            n_nodes = functions.scatter_add(n_nodes, batch,
                                            self.xp.ones(batch.shape[0]))
            y = functions.concat((y, n_nodes.reshape((-1, 1))))

        return y
