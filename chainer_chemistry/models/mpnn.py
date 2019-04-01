from functools import partial
from typing import Optional  # NOQA

import chainer
from chainer import cuda
from chainer import functions
import numpy  # NOQA

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.readout.mpnn_readout import MPNNReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.links.update.mpnn_update import MPNNUpdate
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class MPNN(GraphConvModel):
    """Message Passing Neural Networks (MPNN).

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in
            each layer and the result is concatenated
        weight_tying (bool): enable weight_tying or not
        num_edge_type (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        nn (~chainer.Link): Neural Networks for expanding edge vector
            dimension
        message_func (str): message function. 'edgenet' and 'ggnn' are
            supported.
        readout_func (str): readout function. 'set2set' and 'ggnn' are
            supported.

    """

    def __init__(
            self,
            out_dim,  # type: int
            hidden_channels=16,  # type: int
            n_update_layers=4,  # type: int
            n_atom_types=MAX_ATOMIC_NUM,  # type: int
            concat_hidden=False,  # type: bool
            weight_tying=True,  # type: bool
            n_edge_types=4,  # type: int
            nn=None,  # type: Optional[chainer.Link]
            message_func='edgenet',  # type: str
            readout_func='set2set',  # type: str
            with_gwm=False
    ):
        # type: (...) -> None
        if message_func not in ('edgenet', 'ggnn'):
            raise ValueError(
                'Invalid message function: {}'.format(message_func))
        if readout_func not in ('set2set', 'ggnn'):
            raise ValueError(
                'Invalid readout function: {}'.format(readout_func))

        if message_func == 'edgenet':
            update = MPNNUpdate
        else:
            update = GGNNUpdate

        if readout_func == 'set2set':
            readout = MPNNReadout
        else:
            readout = GGNNReadout

        update_kwargs = {'nn': nn}
        super(MPNN, self).__init__(
            update_layer=update, readout_layer=readout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            weight_tying=weight_tying, n_edge_types=n_edge_types,
            with_gwm=with_gwm, update_kwargs=update_kwargs
        )
        self.message_func = message_func
        self.readout_func = readout_func
