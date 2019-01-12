import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class SchNetReadout(chainer.Chain):
    """SchNet submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each molecule
    """

    def __init__(self, out_dim=1, hidden_dim=32):
        super(SchNetReadout, self).__init__()
        with self.init_scope():
            self.linear1 = GraphLinear(hidden_dim)
            self.linear2 = GraphLinear(out_dim)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

    def __call__(self, h):
        h = self.linear1(h)
        h = functions.softplus(h)
        h = self.linear2(h)
        h = functions.sum(h, axis=1)
        return h
