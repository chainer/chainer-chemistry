"""
Chainer implementation of CFConv.

SchNet: A continuous-filter convolutional neural network for modeling
    quantum interactions
Kristof et al.
See: https://arxiv.org/abs/1706.08566
"""

import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class CFConv(chainer.Chain):
    def __init__(self, num_rbf=300, radius_resolution=0.1, gamma=10.0,
                 hidden_dim=64):
        super(CFConv, self).__init__()
        with self.init_scope():
            self.dense1 = links.Linear(num_rbf, hidden_dim)
            self.dense2 = links.Linear(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.radius_resolution = radius_resolution
        self.gamma = gamma

    def __call__(self, h, dist):
        """
        Args:
            h (numpy.ndarray): axis 0 represents minibatch index,
                axis 1 represents atom_index and axis2 represents
                feature dimension.
            dist (numpy.ndarray): axis 0 represents minibatch index,
                axis 1 and 2 represent distance between atoms.

        """
        mb, atom, ch = h.shape
        if ch != self.hidden_dim:
            raise ValueError('h.shape[2] {} and hidden_dim {} must be same!'
                             .format(ch, self.hidden_dim))
        embedlist = self.xp.arange(
            self.num_rbf).astype('f') * self.radius_resolution
        dist = functions.reshape(dist, (mb, atom, atom, 1))
        dist = functions.broadcast_to(dist, (mb, atom, atom, self.num_rbf))
        dist = functions.exp(- self.gamma * (dist - embedlist) ** 2)
        dist = functions.reshape(dist, (-1, self.num_rbf))
        dist = self.dense1(dist)
        dist = functions.softplus(dist)
        dist = self.dense2(dist)
        dist = functions.softplus(dist)
        dist = functions.reshape(dist, (mb, atom, atom, self.hidden_dim))
        h = functions.reshape(h, (mb, atom, 1, self.hidden_dim))
        h = functions.broadcast_to(h, (mb, atom, atom, self.hidden_dim))
        h = functions.sum(h * dist, axis=1)
        return h


class SchNetUpdate(chainer.Chain):
    def __init__(self, hidden_dim=64):
        super(SchNetUpdate, self).__init__()
        with self.init_scope():
            self.linear = chainer.ChainList(
                *[GraphLinear(hidden_dim) for _ in range(3)])
            self.cfconv = CFConv(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def __call__(self, x, dist):
        v = self.linear[0](x)
        v = self.cfconv(v, dist)
        v = self.linear[1](v)
        v = functions.softplus(v)
        v = self.linear[2](v)
        return x + v
