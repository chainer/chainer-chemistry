from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.schnet_readout import SchNetReadout
from chainer_chemistry.links.update.schnet_update import SchNetUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class SchNet(GraphConvModel):
    """SchNet

    See Kristof et al, \
        SchNet: A continuous-filter convolutional neural network for modeling
        quantum interactions. \
        `arXiv:1706.08566 <https://arxiv.org/abs/1706.08566>`_

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
        num_rbf (int): Number of RDF kernels used in `CFConv`.
        radius_resolution (float): Resolution of radius.
            The range (radius_resolution * 1 ~ radius_resolution * num_rbf)
            are taken inside `CFConv`.
        gamma (float): exponential factor of `CFConv`'s radius kernel.
    """

    def __init__(self, out_dim=1, hidden_channels=64, n_layers=3,
                 readout_hidden_dim=32, n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False, num_rbf=300, radius_resolution=0.1,
                 gamma=10.0, with_gwm=False):
        # TODO: use readout_hidden_dim
        # TODO: use num_rbf, radius_resolution, gamma in update
        readout_kwargs = {'hidden_channels': readout_hidden_dim}
        super(SchNet, self).__init__(
            update_layer=SchNetUpdate, readout_layer=SchNetReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_layers=n_layers, n_atom_types=n_atom_types,
            concat_hidden=concat_hidden, with_gwm=with_gwm,
            readout_kwargs=readout_kwargs
        )
