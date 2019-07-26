# -*- coding: utf-8 -*-
from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.relgat_update import RelGATUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class RelGAT(GraphConvModel):
    """Relational Graph Attention Networks (GAT)

    See: Veličković, Petar, et al. (2017).\
        Graph Attention Networks.\
        `arXiv:1701.10903 <https://arxiv.org/abs/1710.10903>`\
        Dan Busbridge, et al. (2018).\
        Relational Graph Attention Networks
        `<https://openreview.net/forum?id=Bklzkh0qFm>`\


    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        dropout_ratio (float): dropout ratio of the normalized attention
            coefficients
        weight_tying (bool): enable weight_tying or not
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        with_gwm (bool): Use GWM module or not.
        n_heads (int): number of multi-head-attentions.
        negative_slope (float): LeakyRELU angle of the negative slope
        softmax_mode (str): take the softmax over the logits 'across' or
            'within' relation. If you would like to know the detail discussion,
            please refer Relational GAT paper.
        concat_heads (bool) : Whether to concat or average multi-head
            attentions
    """
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 dropout_ratio=-1., weight_tying=False,
                 activation=functions.identity, n_edge_types=4,
                 with_gwm=False, n_heads=3, negative_slope=0.2,
                 softmax_mode='across', concat_heads=False):
        if concat_heads:
            channels = [hidden_channels * n_heads
                        for _ in range(n_update_layers)]
            out_channels = [hidden_channels
                            for _ in range(n_update_layers)]
            channels[0] = hidden_channels
            hidden_channels = channels
        else:
            out_channels = None
        update_kwargs = {'n_heads': n_heads, 'dropout_ratio': dropout_ratio,
                         'negative_slope': negative_slope,
                         'softmax_mode': softmax_mode,
                         'concat_heads': concat_heads}
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}

        super(RelGAT, self).__init__(
            update_layer=RelGATUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, n_update_layers=n_update_layers,
            hidden_channels=hidden_channels,
            out_channels=out_channels, n_atom_types=n_atom_types,
            concat_hidden=concat_hidden, weight_tying=weight_tying,
            dropout_ratio=dropout_ratio, n_edge_types=n_edge_types,
            with_gwm=with_gwm, update_kwargs=update_kwargs,
            readout_kwargs=readout_kwargs
        )
