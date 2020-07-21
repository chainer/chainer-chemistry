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

from chainer_chemistry.functions import shifted_softplus
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class CFConv(chainer.Chain):
    """CFConv

    Args:
        num_rbf (int): Number of RBF kernel
        radius_resolution (float): resolution of radius.
            Roughly `num_rbf * radius_resolution` ball is convolved in 1 step.
        gamma (float): coefficient to apply kernel.
        hidden_dim (int): hidden dim
    """

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
        """main calculation

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
        dist = shifted_softplus(dist)
        dist = self.dense2(dist)
        dist = shifted_softplus(dist)
        dist = functions.reshape(dist, (mb, atom, atom, self.hidden_dim))
        h = functions.reshape(h, (mb, atom, 1, self.hidden_dim))
        h = functions.broadcast_to(h, (mb, atom, atom, self.hidden_dim))
        h = functions.sum(h * dist, axis=1)
        return h


class SchNetUpdate(chainer.Chain):
    """Update submodule for SchNet

    `in_channels` and `hidden_channels` must be same with `hidden_channels` in
     this module.

    Args:
        hidden_channels (int):
        num_rbf (int):
        radius_resolution (float):
        gamma (float):
    """

    def __init__(self, hidden_channels=64, num_rbf=300,
                 radius_resolution=0.1, gamma=10.0):
        super(SchNetUpdate, self).__init__()
        with self.init_scope():
            self.linear = chainer.ChainList(
                *[GraphLinear(None, hidden_channels) for _ in range(3)])
            self.cfconv = CFConv(
                num_rbf=num_rbf, radius_resolution=radius_resolution,
                gamma=gamma, hidden_dim=hidden_channels)
        self.hidden_channels = hidden_channels

    def __call__(self, h, adj, **kwargs):
        v = self.linear[0](h)
        v = self.cfconv(v, adj)
        v = self.linear[1](v)
        v = shifted_softplus(v)
        v = self.linear[2](v)
        return h + v
