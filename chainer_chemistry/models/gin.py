import chainer
from chainer import functions, cuda

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.gin_update import GINUpdate


class GIN(chainer.Chain):
    """Simple implementation of Graph Isomorphism Network (GIN)

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
    """
    def __init__(self, out_dim, hidden_channels=16,
                 n_update_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5, concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 n_edge_types=4):
        super(GIN, self).__init__()
        n_message_layer = 1 if weight_tying else n_update_layers
        n_readout_layer = n_update_layers if concat_hidden else 1
        with self.init_scope():
            # embedding
            self.embed = EmbedAtomID(out_size=hidden_channels,
                                     in_size=n_atom_types)

            # two non-linear MLP part
            self.update_layers = chainer.ChainList(*[GINUpdate(
                hidden_channels=hidden_channels, dropout_ratio=dropout_ratio)
                for _ in range(n_message_layer)])

            # Readout
            self.readout_layers = chainer.ChainList(*[GGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels * 2,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
        # end with

        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.n_message_layers = n_message_layer
        self.n_readout_layer = n_readout_layer
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying
        self.n_edge_types = n_edge_types

    def __call__(self, atom_array, adj, is_real_node=None):
        """forward propagation

        Args:
            atom_array (numpy.ndarray): mol-minibatch by node numpy.ndarray,
                minibatch of molecular which is represented with atom IDs
                (representing C, O, S, ...) atom_array[m, i] = a represents
                m-th molecule's i-th node is value a (atomic number)
            adj (numpy.ndarray): mol-minibatch by relation-types by node by
                node numpy.ndarray,
                minibatch of multple relational adjancency matrix with
                edge-type information adj[i, j] = b represents
                m-th molecule's  edge from node i to node j has value b
            is_real_node:

        Returns:
            numpy.ndarray: final molecule representation
        """

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array

        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)

        g_list = []
        for step in range(self.n_message_layers):
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
