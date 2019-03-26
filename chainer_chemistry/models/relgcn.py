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

    Args:
        out_channels (int): dimension of output feature vector
        num_edge_type (int): number of types of edge
        ch_list (list): channels of each update layer
        n_atom_types (int): number of types of atoms
        input_type (str): type of input vector
        scale_adj (bool): If ``True``, then this network normalizes
            adjacency matrix
    """

    def __init__(self, out_channels=64, n_edge_types=4, ch_list=None,
                 n_atom_types=MAX_ATOMIC_NUM, input_type='int',
                 scale_adj=False, with_gwm=False):
        # TODO: input_type is deprecated
        if ch_list is None:
            ch_list = [16, 128, 64]
        readout_kwargs = {'nobias': True,
                          'activation': functions.tanh}
        super(RelGCN, self).__init__(
            update_layer=RelGCNUpdate, readout_layer=GGNNReadout,
            out_dim=out_channels, n_edge_types=n_edge_types, in_channels=ch_list,
            n_atom_types=n_atom_types, scale_adj=scale_adj, with_gwm=with_gwm,
            activation=functions.tanh, readout_kwargs=readout_kwargs
        )
