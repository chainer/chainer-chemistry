import chainer
from chainer import Variable, functions  # NOQA

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.nfp_readout import NFPReadout
from chainer_chemistry.links.update.nfp_update import NFPUpdate


class NFP(chainer.Chain):
    """Neural Finger Print (NFP)

    See: David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael
        Bombarell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P Adams (2015).
        Convolutional networks on graphs for learning molecular fingerprints.
        *Advances in Neural Information Processing Systems (NIPS) 28*,

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (int): dimension of feature vector for each node
        n_update_layers (int): number of layers
        max_degree (int): max degree of atoms
            when molecules are regarded as graphs
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
    """
    def __init__(self, out_dim, hidden_channels=16, n_update_layers=4,
                 max_degree=6, n_atom_types=MAX_ATOMIC_NUM,
                 concat_hidden=False):
        super(NFP, self).__init__()
        n_degree_types = max_degree + 1
        with self.init_scope():
            self.embed = EmbedAtomID(in_size=n_atom_types,
                                     out_size=hidden_channels)
            self.layers = chainer.ChainList(
                *[NFPUpdate(hidden_channels, hidden_channels,
                            max_degree=max_degree)
                  for _ in range(n_update_layers)])
            self.readout_layers = chainer.ChainList(
                *[NFPReadout(out_dim=out_dim, in_channels=hidden_channels)
                  for _ in range(n_update_layers)])
        self.out_dim = out_dim
        self.hidden_channels = hidden_channels
        self.max_degree = max_degree
        self.n_degree_types = n_degree_types
        self.n_update_layers = n_update_layers
        self.concat_hidden = concat_hidden

    def __call__(self, atom_array, adj, is_real_node=None):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix
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
            for degree in range(1, self.n_degree_types + 1)]
        g_list = []
        for update, readout in zip(self.layers, self.readout_layers):
            h = update(h, adj, deg_conds)
            dg = readout(h, is_real_node)
            g = g + dg
            if self.concat_hidden:
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=2)
        else:
            return g
