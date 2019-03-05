import chainer
from chainer import cuda
from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.gin_readout import GINReadout
from chainer_chemistry.links.update.gin_update import GINUpdate



class GIN(chainer.Chain):
    """
    Simplest implementation of Graph Isomorphism Network (GIN)

    See: Xu, Hu, Leskovec, and Jegelka, "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (default=16): dimension of hidden vectors
            associated to each atom
        n_layers (default=4): number of layers
        n_atom_types: number of atoms
        dropout_ratio (default=0.5); if > 0.0, perform dropout
        concat_hidden (default=False): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (default=True): enable weight_tying for all units


    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 weight_tying=True,
                 activation=functions.identity):
        super(GIN, self).__init__()

        n_message_layer = 1 if weight_tying else n_layers
        n_readout_layer = n_layers if concat_hidden else 1
        with self.init_scope():
            # embedding
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            # two non-linear MLP part
            self.update_layers = chainer.ChainList(*[GINUpdate(
                hidden_dim=hidden_dim, dropout_ratio=dropout_ratio)
                for _ in range(n_message_layer)])

            # Readout
            self.readout_layers = chainer.ChainList(*[GINReadout(
                out_dim=out_dim, hidden_dim=hidden_dim,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
        # end with

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_message_layers = n_message_layer
        self.n_readout_layer = n_readout_layer
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def __call__(self, atom_array, adj, is_real_node=None):
        """
        Describe the whole forwar path

        Args:
            atom_array (numpy.ndarray): mol-minibatch by node numpy.ndarray,
                minibatch of molecular which is represented with atom IDs (representing C, O, S, ...)
                atom_array[m, i] = a represents
                m-th molecule's i-th node is value a (atomic number)
            adj (numpy.ndarray): mol-minibatch by relation-types by node by node numpy.ndarray,
                       minibatch of multple relational adjancency matrix with edge-type information
                       adj[i, j] = b represents
                       m-th molecule's  edge from node i to node j has value b
            is_real_node:

        Returns:
            numpy.ndarray: final molecule representation
        """

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        # end if-else
        # print("for DEBUG: graphtransformer.py::__call__(): xp.shape(h)=" + str(xp.shape(h)))


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
