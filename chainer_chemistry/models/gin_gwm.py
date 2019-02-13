import chainer
from chainer import cuda
from chainer import functions
from chainer import links

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.gin_readout import GINReadout
from chainer_chemistry.links.update.gin_update import GINUpdate
from chainer_chemistry.models.gwm import GWM


class GIN_GWM(chainer.Chain):
    """
    Simplest implementation of Graph Isomorphism Network (GIN) , attache with the Graph Warp Module (GWM)

    See: Xu, Hu, Leskovec, and Jegelka, "How powerful are graph neural networks?", in ICLR 2019.

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module for Boosting the Power of Graph Neural Networks", arXiv, 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of hidden vectors
            associated to each atom
        hiden_dim_super(int); dimension of super-node hidden vector
        n_layers (int): number of layers
        n_heads (int): numbef of heads
        n_atom_types (int): number of atoms
        n_super_feature (int); number of super-node observation attributes
        dropout_ratio (float); if > 0.0, perform dropout
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying for all units


    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16, hidden_dim_super=16,
                 n_layers=4, n_heads=8,n_atom_types=MAX_ATOMIC_NUM,
                 n_super_feature=2 + 2 + MAX_ATOMIC_NUM*2,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 weight_tying=True,
                 activation=functions.identity):
        super(GIN_GWM, self).__init__()

        n_message_layer = 1 if weight_tying else n_layers
        n_readout_layer = n_layers if concat_hidden else 1
        with self.init_scope():
            # embedding
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            # two non-linear MLP part
            self.update_layers = chainer.ChainList(*[GINUpdate(
                hidden_dim=hidden_dim, dropout_ratio=dropout_ratio)
                for _ in range(n_message_layer)])

            # GWM
            self.embed_super = links.Linear(in_size=n_super_feature, out_size=hidden_dim_super)
            self.gwm = GWM(hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                 n_layers=n_message_layer, n_heads=n_heads,
                 dropout_ratio=dropout_ratio,
                 tying_flag=weight_tying,
                 gpu=-1)

            # Readout
            self.readout_layers = chainer.ChainList(*[GINReadout(
                out_dim=out_dim, hidden_dim=hidden_dim,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
            self.linear_for_concat_super = links.Linear(in_size=None, out_size=out_dim)
        # end with

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_message_layers = n_message_layer
        self.n_readout_layer = n_readout_layer
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def __call__(self, atom_array, adj, super_node, is_real_node=None):
        """
        Describe a layer

        Args:
            atom_array (numpy.ndarray): mol-minibatch by node numpy.ndarray,
                minibatch of molecular which is represented with atom IDs (representing C, O, S, ...)
                atom_array[m, i] = a represents
                m-th molecule's i-th node is value a (atomic number)
            adj (numpy.ndarray): mol-minibatch by relation-types by node by node numpy.ndarray,
                       minibatch of multiple relational adjancency matrix with edge-type information
                       adj[i, j] = b represents
                       m-th molecule's  edge from node i to node j has value b
            super_node (numpy.ndarray): 1D array, the supernode hidden state
            is_real_node:

        Returns:
            numpy.ndarray: final molecule representation
        """


        # assert len(super_node) > 0
        # print("for DEBUG: graphtransformer.py::__call__(): len(super_node)=" + str(len(super_node)))

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        # end if-else
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)

        self.gwm.GRU_local.reset_state()
        self.gwm.GRU_super.reset_state()

        # ebmbed super node
        h_s = self.embed_super(super_node)

        g_list = []
        for step in range(self.n_message_layers):
            message_layer_index = 0 if self.weight_tying else step
            h2 = self.update_layers[message_layer_index](h, adj)
            h, h_s = self.gwm(h, h2, h_s, message_layer_index)
            if self.concat_hidden:
                g = self.readout_layers[step](h, h0, is_real_node)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout_layers[0](h, h0, is_real_node)
            g2 = functions.concat( (g, h_s), axis=1 )
            out_g = functions.relu(self.linear_for_concat_super(g2))
            return out_g
