import chainer
from chainer import functions, cuda  # NOQA

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.utils import convert_sparse_with_edge_type


class GGNN(chainer.Chain):
    """Gated Graph Neural Networks (GGNN)

    See: Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015).\
        Gated graph sequence neural networks. \
        `arXiv:1511.05493 <https://arxiv.org/abs/1511.05493>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
    """
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4):
        super(GGNN, self).__init__()
        n_readout_layer = n_update_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_update_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(
                out_size=hidden_channels, in_size=n_atom_types)
            self.update_layers = chainer.ChainList(*[GGNNUpdate(
                hidden_channels=hidden_channels, n_edge_types=n_edge_types)
                for _ in range(n_message_layer)])
            # Readout
            self.readout_layers = chainer.ChainList(*[GGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels * 2,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.n_update_layers = n_update_layers
        self.n_edge_types = n_edge_types
        self.activation = activation
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def __call__(self, atom_array, adj, is_real_node=None):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information
            is_real_node (numpy.ndarray): 2-dim array (minibatch, num_nodes).
                1 for real node, 0 for virtual node.
                If `None`, all node is considered as real node.
        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # reset state
        self.reset_state()
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        for step in range(self.n_update_layers):
            message_layer_index = 0 if self.weight_tying else step
            h = self.update_layers[message_layer_index](h, adj)
            if self.concat_hidden:
                g = self.readout_layers[step](h, h0, is_real_node)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout_layers[0](h, h0, is_real_node)
            return g

    def reset_state(self):
        [update_layer.reset_state() for update_layer in self.update_layers]


class SparseGGNN(GGNN):
    """GGNN model for sparse matrix inputs.

    The constructor of this model is the same with that of GGNN.
    See the documentation of GGNN for the detail.
    """

    def __init__(self, *args, **kwargs):
        super(SparseGGNN, self).__init__(*args, **kwargs)

    def __call__(self, atom_array, data, row, col, edge_type,
                 is_real_node=None):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            data (numpy.ndarray): the entries of the batched sparse matrix.
            row (numpy.ndarray): the row indices of the matrix entries.
            col (numpy.ndarray): the column indices of the matrix entries.
            edge_type (numpy.ndarray): edge type information of edges.
            is_real_node (numpy.ndarray): 2-dim array (minibatch, num_nodes).
                1 for real node, 0 for virtual node.
                If `None`, all node is considered as real node.

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        num_nodes = atom_array.shape[1]
        adj = convert_sparse_with_edge_type(
            data, row, col, num_nodes, edge_type, self.n_edge_types)
        return super(SparseGGNN, self).__call__(
            atom_array, adj, is_real_node=is_real_node)
