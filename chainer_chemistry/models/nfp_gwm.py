import chainer
from chainer import functions
from chainer import links
from chainer import Variable

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.readout.nfp_readout import NFPReadout
from chainer_chemistry.links.update.nfp_update import NFPUpdate
from chainer_chemistry.models.gwm import GWM


class NFP_GWM(chainer.Chain):

    """Neural Finger Print (NFP) with Graph Warp Module (GWM)

    See: David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael
        Bombarell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P Adams (2015).
        Convolutional networks on graphs for learning molecular fingerprints.
        *Advances in Neural Information Processing Systems (NIPS) 28*,

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module for Boosting the Power of Graph Neural Networks", arXiv, 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        hidden_dim_super(default=16); dimension of super-node hidden vector
        n_layers (int): number of layers
        n_heads (default=8): numbef of heads
        max_degree (int): max degree of atoms
            when molecules are regarded as graphs
        n_atom_types (int): number of types of atoms
        n_super_feature (int); number of super-node observation attributes
        dropout_ratio (float); if > 0.0, perform dropout
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated

    """

    def __init__(self, out_dim, hidden_dim=16, hidden_dim_super=16, n_layers=4, n_heads=8,
                 max_degree=6, n_atom_types=MAX_ATOMIC_NUM,
                 n_super_feature=2 + 2 + MAX_ATOMIC_NUM*2,
                 dropout_ratio=0.5,
                 concat_hidden=False):
        super(NFP_GWM, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.embed = EmbedAtomID(in_size=n_atom_types, out_size=hidden_dim)
            self.layers = chainer.ChainList(
                *[NFPUpdate(hidden_dim, hidden_dim, max_degree=max_degree)
                  for _ in range(n_layers)])

            # GWM
            self.embed_super = links.Linear(in_size=n_super_feature, out_size=hidden_dim_super)
            self.gwm = GWM(hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                 n_layers=n_layers, n_heads=n_heads,
                 dropout_ratio=dropout_ratio,
                 tying_flag=False,
                 gpu=-1)

            self.read_out_layers = chainer.ChainList(
                *[NFPReadout(hidden_dim, out_dim)
                  for _ in range(n_layers)])
            self.linear_for_concat_super = links.Linear(in_size=None, out_size=out_dim)
        # end with
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.max_degree = max_degree
        self.num_degree_type = num_degree_type
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden

    def __call__(self, atom_array, adj, super_node, is_real_node=None):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjacency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix
            super_node (numpy.ndarray): 1D array, the super-node observation.
            is_real_node (numpy.ndarray): 2-dim array (minibatch, num_nodes).
                1 for real node, 0 for virtual node.
                If `None`, all node is considered as real node.

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        if atom_array.dtype == self.xp.int32:
            # atom_array: (minibatch, atom)
            h = self.embed(atom_array)
        else:
            h = atom_array
        # h: (minibatch, atom, ch)
        g = 0

        self.gwm.GRU_local.reset_state()
        self.gwm.GRU_super.reset_state()

        # ebmbed super node
        h_s = self.embed_super(super_node)

        # --- NFP update & readout ---
        # degree_mat: (minibatch, max_num_atoms)
        if isinstance(adj, Variable):
            adj_array = adj.data
        else:
            adj_array = adj
        degree_mat = self.xp.sum(adj_array, axis=1)
        # deg_conds: (minibatch, atom, ch)
        deg_conds = [self.xp.broadcast_to(
            ((degree_mat - degree) == 0)[:, :, None], h.shape)
            for degree in range(1, self.num_degree_type + 1)]
        g_list = []
        layer = 0
        for update, readout in zip(self.layers, self.read_out_layers):
            #h = update(h, adj, deg_conds)
            h2 = update(h, adj, deg_conds)
            h, h_s = self.gwm(h, h2, h_s, layer)

            dg = readout(h, is_real_node)
            g = g + dg
            if self.concat_hidden:
                g_list.append(g)
            layer = layer + 1

        if self.concat_hidden:
            return functions.concat(g_list, axis=2)
        else:
            g2 = functions.concat( (g, h_s), axis=1 )
            out_g = functions.relu(self.linear_for_concat_super(g2))
            return out_g
