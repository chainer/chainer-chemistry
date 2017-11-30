"""
Implementation of Neural Fingerprint

"""
import chainer
from chainer import cuda
from chainer import functions
import numpy

import chainerchem
from chainerchem.config import MAX_ATOMIC_NUM


class NFPUpdate(chainer.Chain):

    def __init__(self, max_degree, hidden_dim, out_dim):
        super(NFPUpdate, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.hidden_weights = chainer.ChainList(
                *[chainerchem.links.GraphLinear(hidden_dim, hidden_dim)
                  for _ in range(num_degree_type)]
            )
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def __call__(self, h, adj, deg_conds):
        # h    (mb, ch, atom)
        # h is used for keep each atom's info for next step -> hidden_dim
        # adjs (mb, edge_type, atom, atom)

        # --- Message part ---
        # Take sum along adjacent atoms

        # fv   (mb, ch, atom)
        fv = chainerchem.functions.matmul(h, adj)

        # --- Update part ---
        # s0, s1, s2 = fv.shape
        if self.xp is numpy:
            zero_array = numpy.zeros(fv.shape, dtype=numpy.float32)
        else:
            zero_array = self.xp.zeros_like(fv)

        fvds = [functions.where(cond, fv, zero_array) for cond in deg_conds]

        out_h = 0
        for hidden_weight, fvd in zip(self.hidden_weights, fvds):
            out_h = out_h + hidden_weight(fvd)

        # out_x shape (minibatch, max_num_atoms, hidden_dim)
        out_h = functions.sigmoid(out_h)
        return out_h


class NFPReadout(chainer.Chain):

    def __init__(self, hidden_dim, out_dim):
        super(NFPReadout, self).__init__()
        with self.init_scope():
            self.output_weight = chainerchem.links.GraphLinear(
                hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def __call__(self, h):
        # input  h shape (mb, ch, atom)
        # return i shape (mb, ch)

        # --- Readout part ---
        i = self.output_weight(h)
        i = functions.softmax(i)
        i = functions.sum(i, axis=2)  # sum along atom's axis
        return i


class NFP(chainer.Chain):

    """Neural Finger Print (NFP)

    Args:
        hidden_dim (int): dimension of feature vector
            associated to each atom
        out_dim (int): dimension of output feature vector
        max_degree (int): max degree of atoms
            when molecules are regarded as graphs
        n_atom_types (int): number of types of atoms
        n_layer (int): number of layers
    """

    def __init__(self, hidden_dim, out_dim, n_layers, max_degree=6,
                 n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False):
        super(NFP, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.embed = chainerchem.links.EmbedAtomID(
                n_atom_types, hidden_dim)
            self.layers = chainer.ChainList(
                *[NFPUpdate(max_degree, hidden_dim, out_dim)
                  for _ in range(n_layers)])
            self.read_out_layers = chainer.ChainList(
                *[NFPReadout(hidden_dim, out_dim)
                  for _ in range(n_layers)])
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.num_degree_type = num_degree_type
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden

    def __call__(self, atom_array, adj):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # TODO(Nakago): update implementation
        if atom_array.dtype == numpy.int32 or \
                atom_array.dtype == cuda.cupy.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        # x: (minibatch, hidden_dim, max_num_atoms)
        g = 0

        # --- NFP update & readout ---
        # degree_mat: (minibatch, max_num_atoms)
        degree_mat = functions.sum(adj, axis=1)
        deg_conds = [self.xp.broadcast_to(
            ((degree_mat - degree).data == 0)[:, None, :], h.shape)
            for degree in range(1, self.num_degree_type + 1)]
        g_list = []
        for update, readout in zip(self.layers, self.read_out_layers):
            h = update(h, adj, deg_conds)
            dg = readout(h)
            g = g + dg
            if self.concat_hidden:
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            return g
