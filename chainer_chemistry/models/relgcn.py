from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.relgcn_update import RelGCNUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class RelGCN(GraphConvModel):

    """Relational GCN (RelGCN)

    See: Michael Schlichtkrull+, \
        Modeling Relational Data with Graph Convolutional Networks. \
        March 2017. \
        `arXiv:1703.06103 <https://arxiv.org/abs/1703.06103>`

    # TODO: `input_type` support is dropped to unify graph conv model.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (None or int or list):
            dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        scale_adj (bool): If ``True``, then this network normalizes
            adjacency matrix
        with_gwm (bool): Use GWM module or not.
    """

    def __init__(self, out_dim=64, hidden_channels=None, n_update_layers=None,
                 n_atom_types=MAX_ATOMIC_NUM, n_edge_types=4,
                 scale_adj=False, with_gwm=False):
        if hidden_channels is None:
            hidden_channels = [16, 128, 64]
        readout_kwargs = {'nobias': True,
                          'activation': functions.tanh}
        super(RelGCN, self).__init__(
            update_layer=RelGCNUpdate, readout_layer=GGNNReadout,
            hidden_channels=hidden_channels,
            out_dim=out_dim, n_edge_types=n_edge_types,
            n_update_layers=n_update_layers, n_atom_types=n_atom_types,
            scale_adj=scale_adj, with_gwm=with_gwm,
            activation=functions.tanh, readout_kwargs=readout_kwargs
        )
