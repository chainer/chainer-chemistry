import chainer
from chainer import functions
from chainer import Variable

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.links.update.rsgcn_update import RSGCNUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class RSGCN(GraphConvModel):

    """Renormalized Spectral Graph Convolutional Network (RSGCN)

    See: Thomas N. Kipf and Max Welling, \
        Semi-Supervised Classification with Graph Convolutional Networks. \
        September 2016. \
        `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_

    The name of this model "Renormalized Spectral Graph Convolutional Network
    (RSGCN)" is named by us rather than the authors of the paper above.
    The authors call this model just "Graph Convolution Network (GCN)", but
    we think that "GCN" is bit too general and may cause namespace issue.
    That is why we did not name this model as GCN.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        use_batch_norm (bool): If True, batch normalization is applied after
            graph convolution.
        readout (Callable): readout function. If None,
            `GeneralReadout(mode='sum)` is used.
            To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
        dropout_ratio (float): ratio used in dropout function.
            If 0 or negative value is set, dropout function is skipped.
        with_gwm (bool): Use GWM module or not.
    """

    def __init__(self, out_dim, hidden_channels=32, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None, dropout_ratio=0.5,
                 with_gwm=False):
        if readout is None:
            readout = GeneralReadout
        super(RSGCN, self).__init__(
            update_layer=RSGCNUpdate, readout_layer=readout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers, n_atom_types=n_atom_types,
            use_batchnorm=use_batch_norm, activation=functions.relu,
            n_activation=n_update_layers-1, dropout_ratio=dropout_ratio,
            with_gwm=with_gwm
        )
