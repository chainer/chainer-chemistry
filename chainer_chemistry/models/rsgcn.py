"""
Implementation of Renormalized Spectral Graph Convolutional Network (RSGCN)

See: Thomas N. Kipf and Max Welling, \
    Semi-Supervised Classification with Graph Convolutional Networks. \
    September 2016. \
    `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_
"""
import chainer
from chainer import functions
from chainer import Variable

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.links.update.rsgcn_update import RSGCNUpdate


class RSGCN(chainer.Chain):

    """Renormalized Spectral Graph Convolutional Network (RSGCN)

    See: Thomas N. Kipf and Max Welling, \
        Semi-Supervised Classification with Graph Convolutional Networks. \
        September 2016. \
        `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_

    The name of this model "Renormalized Spectral Graph Convolutional Network
    (RSGCN)" is named by us rather than the authors of the paper above.
    The authors call this model just "Graph Convolution Network (GCN)", but
    we think that "GCN" is bit too general and may cause namespace issue.
    That is why we did not name this model as GCN.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_atom_types (int): number of types of atoms
        n_layers (int): number of layers
        use_batch_norm (bool): If True, batch normalization is applied after
            graph convolution.
        readout (Callable): readout function. If None,
            `GeneralReadout(mode='sum)` is used.
            To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
        dropout_ratio (float): ratio used in dropout function.
            If 0 or negative value is set, dropout function is skipped.

    """

    def __init__(self, out_dim, hidden_dim=32, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None, dropout_ratio=0.5):
        super(RSGCN, self).__init__()
        in_dims = [hidden_dim for _ in range(n_layers)]
        out_dims = [hidden_dim for _ in range(n_layers)]
        out_dims[n_layers - 1] = out_dim
        if readout is None:
            readout = GeneralReadout()
        with self.init_scope():
            self.embed = chainer_chemistry.links.EmbedAtomID(
                in_size=n_atom_types, out_size=hidden_dim)
            self.gconvs = chainer.ChainList(
                *[RSGCNUpdate(in_dims[i], out_dims[i])
                  for i in range(n_layers)])
            if use_batch_norm:
                self.bnorms = chainer.ChainList(
                    *[chainer_chemistry.links.GraphBatchNormalization(
                        out_dims[i]) for i in range(n_layers)])
            else:
                self.bnorms = [None for _ in range(n_layers)]
            if isinstance(readout, chainer.Link):
                self.readout = readout
        if not isinstance(readout, chainer.Link):
            self.readout = readout
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio

    def __call__(self, graph, adj):
        """Forward propagation

        Args:
            graph (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        if graph.dtype == self.xp.int32:
            # atom_array: (minibatch, nodes)
            h = self.embed(graph)
        else:
            h = graph
        # h: (minibatch, nodes, ch)

        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)

        # --- RSGCN update ---
        for i, (gconv, bnorm) in enumerate(zip(self.gconvs,
                                               self.bnorms)):
            h = gconv(h, w_adj)
            if bnorm is not None:
                h = bnorm(h)
            if self.dropout_ratio > 0.:
                h = functions.dropout(h, ratio=self.dropout_ratio)
            if i < self.n_layers - 1:
                h = functions.relu(h)

        # --- readout ---
        y = self.readout(h)
        return y
