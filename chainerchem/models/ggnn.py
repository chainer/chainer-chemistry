import chainer
from chainer import cuda
from chainer import functions
from chainer import links
import numpy

import chainerchem
from chainerchem.config import MAX_ATOMIC_NUM
from chainerchem.links import EmbedAtomID
from chainerchem.links import GraphLinear


class GGNN(chainer.Chain):
    """Gated Graph Neural Networks (GG-NN) Li et al.

    Args:
        hidden_dim (int): dimension of feature vector
            associated to each atom
        out_dim (int): dimension of output feature vector
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not
    """
    NUM_EDGE_TYPE = 4

    def __init__(self, hidden_dim, out_dim,
                 n_layers, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True):
        super(GGNN, self).__init__()
        n_readout_layer = 1 if concat_hidden else n_layers
        n_message_layer = 1 if weight_tying else n_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(n_atom_types, hidden_dim)
            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
                  for _ in range(n_message_layer)]
            )
            self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def update(self, h, adj, step=0):
        # --- Message & Update part ---
        # (mb, ch, atom)
        mb, ch, atom = h.shape
        out_ch = ch
        message_layer_index = 0 if self.weight_tying else step
        m = functions.reshape(self.message_layers[message_layer_index](
            h), (mb, out_ch, atom, self.NUM_EDGE_TYPE))
        # (mb, ch, atom, edge_type)
        # Transpose
        m = functions.transpose(m, (0, 3, 1, 2))
        # (mb, edge_type, ch, atom)

        adj = functions.reshape(adj, (mb * self.NUM_EDGE_TYPE, atom, atom))
        # (mb * edge_type, out_ch, atom)
        m = functions.reshape(m, (mb * self.NUM_EDGE_TYPE, out_ch, atom))

        m = chainerchem.functions.matmul(m, adj)

        # (mb * edge_type, out_ch, atom)
        m = functions.reshape(m, (mb, self.NUM_EDGE_TYPE, out_ch, atom))
        # Take sum
        m = functions.sum(m, axis=1)
        # (mb, out_ch, atom)

        # --- Update part ---
        # Contraction
        h = functions.reshape(functions.transpose(h, (0, 2, 1)),
                              (mb * atom, ch))
        # Contraction
        m = functions.reshape(functions.transpose(m, (0, 2, 1)),
                              (mb * atom, ch))
        out_h = self.update_layer(functions.concat((h, m), axis=1))
        # Expansion
        out_h = functions.transpose(functions.reshape(out_h, (mb, atom, ch)),
                                    (0, 2, 1))
        return out_h

    def readout(self, h, h0, step=0):
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        g = functions.sigmoid(
            self.i_layers[index](functions.concat((h, h0), axis=1))) \
            * self.j_layers[index](h)
        g = functions.sum(g, axis=2)  # sum along atom's axis
        return g

    def __call__(self, atom_array, adj):
        """Forward propagation

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
        self.update_layer.reset_state()
        if atom_array.dtype == numpy.int32 \
                or atom_array.dtype == cuda.cupy.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        for step in range(self.n_layers):
            h = self.update(h, adj, step)
            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout(h, h0, 0)
            return g
