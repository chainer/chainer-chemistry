import chainer
from chainer.functions import relu

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GraphMLP(chainer.Chain):

    """Graph MLP layer

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        activation (chainer.functions): activation function
    """

    def __init__(self, out_dim, hidden_dim=16, n_layers=2, activation=relu):
        super(GraphMLP, self).__init__()
        if n_layers <= 0:
            raise ValueError('n_layers must be a positive integer, but it was '
                             'set to {}'.format(n_layers))
        layers = [GraphLinear(None, hidden_dim) for i in range(n_layers - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
            self.l_out = GraphLinear(None, out_dim)
        self.activation = activation

    def __call__(self, x):
        h = x
        for l in self.layers:
            h = self.activation(l(h))
        h = self.l_out(h)
        return h
