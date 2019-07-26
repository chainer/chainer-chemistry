from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.gin_update import GINUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class GIN(GraphConvModel):
    """
    Simplest implementation of Graph Isomorphism Network (GIN)

    See: Xu, Hu, Leskovec, and Jegelka, \
    "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        dropout_ratio (float): dropout ratio. Negative value indicates not
            apply dropout
        weight_tying (bool): enable weight_tying or not
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        with_gwm (bool): Use GWM module or not.
    """
    def __init__(self, out_dim, hidden_channels=16,
                 n_update_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_gwm=False):
        update_kwargs = {'dropout_ratio': dropout_ratio}
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(GIN, self).__init__(
            update_layer=GINUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers, n_atom_types=n_atom_types,
            concat_hidden=concat_hidden, weight_tying=weight_tying,
            n_edge_types=n_edge_types, with_gwm=with_gwm,
            update_kwargs=update_kwargs, readout_kwargs=readout_kwargs
        )
