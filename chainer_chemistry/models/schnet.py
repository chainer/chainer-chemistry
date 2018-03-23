"""
Chainer implementation of SchNet

SchNet: A continuous-filter convolutional neural network for modeling quantum
interactions
Kristof et al.
See: https://arxiv.org/abs/1706.08566
"""
import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.config import MAX_ATOMIC_NUM


class CFConvLayer(chainer.Chain):
    def __init__(self, num_rbf=300, radius_resolution=0.1, gamma=10.0,
                 hidden_dim=64):
        super(CFConvLayer, self).__init__()
        with self.init_scope():
            self.dense1 = links.Linear(num_rbf, hidden_dim)
            self.dense2 = links.Linear(hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.radius_resolution = radius_resolution
        self.gamma = gamma

    def __call__(self, x, r):
        """Forward propagaion

        Args:
            x (numpy.ndarray): axis 0 represents minibatch index,
                axis 1 represents atom_index and axis 2 represents feature
                dimension.
            r (numpy.ndarray): axis 0 represents minibatch index,
                axis 1 and 2 represent distance between atoms.

        """
        # s0 minibatch, s1 atom (from), s2 atom (to)
        s0, s1, s2 = r.shape
        # a0 minibatch, a1 atom, a2 ch. a2 must be equal to self.hidden_dim
        a0, a1, a2 = x.shape
        if a2 != self.hidden_dim:
            raise ValueError("x.shape[2] {} and hidden_dim {} must be same!"
                             .format(a2, self.hidden_dim))
        embedlist = self.xp.arange(
            self.num_rbf, dtype=self.xp.float32) * self.radius_resolution
        r = functions.reshape(r, (s0, s1, s2, 1))
        r = functions.broadcast_to(r, (s0, s1, s2, self.num_rbf))
        r = functions.exp(- self.gamma * (r - embedlist) ** 2)
        r = functions.reshape(r, (s0 * s1 * s2, self.num_rbf))
        r = self.dense1(r)
        r = functions.softplus(r)
        r = self.dense2(r)
        r = functions.softplus(r)
        r = functions.reshape(r, (s0, s1, s2, self.hidden_dim))
        x = functions.reshape(x, (a0, a1, 1, self.hidden_dim))
        x = functions.broadcast_to(x, (a0, a1, s2, self.hidden_dim))
        x = functions.sum(x * r, axis=1)
        return x


class InteractionLayer(chainer.Chain):
    def __init__(self, hidden_dim=64):
        super(InteractionLayer, self).__init__()
        with self.init_scope():
            self.awlayer1 = AtomwiseLinear(hidden_dim)
            self.awlayer2 = AtomwiseLinear(hidden_dim)
            self.awlayer3 = AtomwiseLinear(hidden_dim)
            self.cfconv = CFConvLayer(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def __call__(self, x, r):
        v = self.awlayer1(x)
        v = self.cfconv(v, r)
        v = self.awlayer2(v)
        v = functions.softplus(v)
        v = self.awlayer3(v)
        return x + v


class AtomwiseLinear(chainer.Chain):

    """AtomwiseLinear

    Its functionality concept is same with GraphLinear, but the input
    variable's shape is different. See the comment of `__call__` method.

    """

    def __init__(self, out_dim):
        super(AtomwiseLinear, self).__init__()
        with self.init_scope():
            self.out_dim = out_dim
            self.linear = links.Linear(self.out_dim)

    def __call__(self, embeded_atom_features):
        # s0 is the minibatch axis
        # s1 is the atom axis
        # s2 is the channel (feature) axis
        s0, s1, s2 = embeded_atom_features.shape
        x = functions.reshape(embeded_atom_features, (s0 * s1, s2))
        x = self.linear(x)
        x = functions.reshape(x, (s0, s1, self.out_dim))
        return x


class SchNet(chainer.Chain):
    """SchNet

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        readout_hidden_dim (int): dimension of feature vector
            associated to each molecule
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
    """

    def __init__(self, out_dim=1, hidden_dim=64, n_layers=3,
                 readout_hidden_dim=32, n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False):
        super(SchNet, self).__init__()
        with self.init_scope():
            self.embed = links.EmbedID(n_atom_types, hidden_dim)
            self.awlayer1 = AtomwiseLinear(readout_hidden_dim)
            self.awlayer2 = AtomwiseLinear(out_dim)
            self.i_layers = chainer.ChainList(
                *[InteractionLayer(hidden_dim) for _ in range(n_layers)])
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.readout_hidden_dim = readout_hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden

    def __call__(self, atom_features, dist_features):
        x = self.embed(atom_features)
        h = []
        # --- update part ---
        for i in range(self.n_layers):
            x = self.i_layers[i](x, dist_features)
            if self.concat_hidden:
                h.append(x)
        # --- readout part ---
        if self.concat_hidden:
            x = functions.concat(h, axis=2)
        x = self.awlayer1(x)
        x = functions.softplus(x)
        x = self.awlayer2(x)
        x = functions.sum(x, axis=1)
        return x
