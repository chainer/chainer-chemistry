from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.nfp_readout import NFPReadout
from chainer_chemistry.links.update.nfp_update import NFPUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class NFP(GraphConvModel):
    """Neural Finger Print (NFP)

    See: David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael
        Bombarell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P Adams (2015).
        Convolutional networks on graphs for learning molecular fingerprints.
        *Advances in Neural Information Processing Systems (NIPS) 28*,

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        max_degree (int): max degree of atoms
            when molecules are regarded as graphs
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated

    """
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4, max_degree=6,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False, with_gwm=False):
        update_kwargs = {'max_degree': max_degree}
        super(NFP, self).__init__(
            update_layer=NFPUpdate, readout_layer=NFPReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers, max_degree=max_degree,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            sum_hidden=True, with_gwm=with_gwm, update_kwargs=update_kwargs
        )
        self.max_degree = max_degree
