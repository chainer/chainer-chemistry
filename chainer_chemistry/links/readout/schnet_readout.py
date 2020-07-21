import chainer
from chainer import functions

from chainer_chemistry.functions import shifted_softplus
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class SchNetReadout(chainer.Chain):
    """SchNet submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        in_channels (int or None): dimension of feature vector for each node
        hidden_channels (int): dimension of feature vector for each node
    """

    def __init__(self, out_dim=1, in_channels=None,
                 hidden_channels=32):
        super(SchNetReadout, self).__init__()
        with self.init_scope():
            self.linear1 = GraphLinear(in_channels, hidden_channels)
            self.linear2 = GraphLinear(hidden_channels, out_dim)
        self.out_dim = out_dim
        self.hidden_dim = in_channels

    def __call__(self, h, **kwargs):
        h = self.linear1(h)
        h = shifted_softplus(h)
        h = self.linear2(h)
        h = functions.sum(h, axis=1)
        return h
