from functools import partial
# from typing import Optional  # NOQA

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


class MPNN(chainer.Chain):
    """Message Passing Neural Networks (MPNN).

    """

    def __init__(
            self,
            out_dim,  # type: int
            hidden_dim=16,  # type: int
            n_layers=4,  # type: int
            n_atom_types=MAX_ATOMIC_NUM,  # type: int
            concat_hidden=False,  # type: bool
            weight_tying=True,  # type: bool
            num_edge_type=4,  # type: int
            nn=None,  # type: Optional[chainer.Link]
            message_func='edgenet',  # type: str
            readout_func='set2set'  # type: str
    ):
        # type: (...) -> None
        super(MPNN, self).__init__()
        if message_func not in ('edgenet', 'ggnn'):
            raise ValueError(
                'Invalid message function: {}'.format(message_func))
        if readout_func not in ('set2set', 'ggnn'):
            raise ValueError(
                'Invalid readout function: {}'.format(readout_func))
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            if message_func == 'ggnn':
                self.update_layers = chainer.ChainList(*[
                    GGNNUpdate(
                        hidden_dim=hidden_dim, num_edge_type=num_edge_type)
                    for _ in range(n_message_layer)
                ])
            else:
                self.update_layers = chainer.ChainList(*[
                    MPNNUpdate(hidden_dim=hidden_dim, nn=nn)
                    for _ in range(n_message_layer)
                ])

            # Readout
            if readout_func == 'ggnn':
                self.readout_layers = chainer.ChainList(*[
                    GGNNReadout(out_dim=out_dim, hidden_dim=hidden_dim)
                    for _ in range(n_readout_layer)
                ])
            else:
                self.readout_layers = chainer.ChainList(*[
                    MPNNReadout(
                        out_dim=out_dim, hidden_dim=hidden_dim, n_layers=1)
                    for _ in range(n_readout_layer)
                ])
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_edge_type = num_edge_type
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying
        self.message_func = message_func
        self.readout_func = readout_func

    def __call__(self, atom_array, adj):
        # type: (numpy.ndarray, numpy.ndarray) -> chainer.Variable
        """Forward propagation.

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information

        Returns:
            ~chainer.Variable: minibatch of fingerprint

        """
        # reset state
        self.reset_state()
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)
        else:
            h = atom_array
        if self.readout_func == 'ggnn':
            h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
            readout_layers = [
                partial(readout_layer, h0=h0)
                for readout_layer in self.readout_layers
            ]
        else:
            readout_layers = self.readout_layers
        g_list = []
        for step in range(self.n_layers):
            message_layer_index = 0 if self.weight_tying else step
            h = self.update_layers[message_layer_index](h, adj)
            if self.concat_hidden:
                g = readout_layers[step](h)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = readout_layers[0](h)
            return g

    def reset_state(self):
        # type: () -> None
        [update_layer.reset_state() for update_layer in self.update_layers]
