from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.utils import convert_sparse_with_edge_type
from chainer_chemistry.models.graph_conv_model import GraphConvModel


class GGNN(GraphConvModel):
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
        with_gwm (bool): Use GWM module or not.
    """
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4, with_gwm=False):
        readout_kwargs = {'activation': activation,
                          'activation_agg': activation}
        super(GGNN, self).__init__(
            update_layer=GGNNUpdate, readout_layer=GGNNReadout,
            out_dim=out_dim, hidden_channels=hidden_channels,
            n_update_layers=n_update_layers,
            n_atom_types=n_atom_types, concat_hidden=concat_hidden,
            weight_tying=weight_tying, n_edge_types=n_edge_types,
            with_gwm=with_gwm, readout_kwargs=readout_kwargs
        )


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
