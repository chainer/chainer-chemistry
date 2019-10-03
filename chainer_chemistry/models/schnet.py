import chainer
from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.schnet_readout import SchNetReadout
from chainer_chemistry.links.update.schnet_update import SchNetUpdate


class SchNet(chainer.Chain):
    """SchNet

    See Kristof et al, \
        SchNet: A continuous-filter convolutional neural network for modeling
        quantum interactions. \
        `arXiv:1706.08566 <https://arxiv.org/abs/1706.08566>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        readout_hidden_dim (int): dimension of feature vector
            associated to each molecule
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        num_rbf (int): Number of RDF kernels used in `CFConv`.
        radius_resolution (float): Resolution of radius.
            The range (radius_resolution * 1 ~ radius_resolution * num_rbf)
            are taken inside `CFConv`.
        gamma (float): exponential factor of `CFConv`'s radius kernel.
    """

    def __init__(self, out_dim=1, hidden_channels=64, n_update_layers=3,
                 readout_hidden_dim=32, n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False, num_rbf=300, radius_resolution=0.1,
                 gamma=10.0):
        super(SchNet, self).__init__()
        with self.init_scope():
            self.embed = EmbedAtomID(out_size=hidden_channels,
                                     in_size=n_atom_types)
            self.update_layers = chainer.ChainList(
                *[SchNetUpdate(
                    hidden_channels,
                    num_rbf=num_rbf, radius_resolution=radius_resolution,
                    gamma=gamma) for _ in range(n_update_layers)])
            self.readout_layer = SchNetReadout(
                out_dim, in_channels=None, hidden_channels=readout_hidden_dim)
        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.readout_hidden_dim = readout_hidden_dim
        self.n_update_layers = n_update_layers
        self.concat_hidden = concat_hidden

    def __call__(self, atom_features, dist_features):
        x = self.embed(atom_features)
        h = []
        # --- update part ---
        for i in range(self.n_update_layers):
            x = self.update_layers[i](x, dist_features)
            if self.concat_hidden:
                h.append(x)
        # --- readout part ---
        if self.concat_hidden:
            x = functions.concat(h, axis=2)
        x = self.readout_layer(x)
        return x
