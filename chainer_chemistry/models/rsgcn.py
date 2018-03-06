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


class RSGCNUpdate(chainer.Chain):
    """RSGCN sub module for message and update part

    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
    """

    def __init__(self, in_channels, out_channels):
        super(RSGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear = chainer_chemistry.links.GraphLinear(
                in_channels, out_channels, nobias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, x, w_adj):
        # --- Message part ---
        h = chainer_chemistry.functions.matmul(w_adj, x)
        # --- Update part ---
        y = self.graph_linear(h)
        return y


class SparseRSGCNUpdate(RSGCNUpdate):
    """Sparse RSGCN sub module for message and update part"""

    def __call__(self, x, sp_adj):
        # --- Message part ---
        h = chainer_chemistry.functions.sparse_matmul(sp_adj, x)
        # --- Update part ---
        y = self.graph_linear(h)
        return y


def rsgcn_readout_sum(x, activation=None):
    """Default readout function for `RSGCN`

    Args:
        x (chainer.Variable): shape consists of (minibatch, atom, ch).
        activation: activation function, default is `None`.
            You may consider taking other activations, for example `sigmoid`,
            `relu` or `softmax` along `axis=2` (ch axis) etc.
    Returns: result of readout, its shape should be (minibatch, out_ch)

    """
    if activation is not None:
        h = activation(x)
    else:
        h = x
    y = functions.sum(h, axis=1)  # sum along node axis
    return y


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
        readout (Callable): readout function. If None, `rsgcn_readout_sum` is
            used. To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
    """

    def __init__(self, out_dim, hidden_dim=32, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None):
        super(RSGCN, self).__init__()
        self.in_dims = [hidden_dim for _ in range(n_layers)]
        self.out_dims = [hidden_dim for _ in range(n_layers)]
        self.out_dims[n_layers - 1] = out_dim
        self.readout = None
        with self.init_scope():
            self.embed = chainer_chemistry.links.EmbedAtomID(
                in_size=n_atom_types, out_size=hidden_dim)
            if getattr(self, '_use_sparse_matrix', False):
                self.gconvs = chainer.ChainList(
                    *[SparseRSGCNUpdate(self.in_dims[i], self.out_dims[i])
                      for i in range(n_layers)])
            else:
                self.gconvs = chainer.ChainList(
                    *[RSGCNUpdate(self.in_dims[i], self.out_dims[i])
                      for i in range(n_layers)])
            if use_batch_norm:
                self.bnorms = chainer.ChainList(
                    *[chainer_chemistry.links.GraphBatchNormalization(
                        self.out_dims[i]) for i in range(n_layers)])
            else:
                self.bnorms = [None for _ in range(n_layers)]
            if isinstance(readout, chainer.Link):
                self.readout = readout
        if self.readout is None:
            self.readout = readout or rsgcn_readout_sum
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _forward_core(self, graph, adj):
        if graph.dtype == self.xp.int32:
            # atom_array: (minibatch, nodes)
            h = self.embed(graph)
        else:
            h = graph

        for i, (gconv, bnorm) in enumerate(zip(self.gconvs,
                                               self.bnorms)):
            h = gconv(h, adj)
            if bnorm is not None:
                h = bnorm(h)
            h = functions.dropout(h)
            if i < self.n_layers - 1:
                h = functions.relu(h)

        y = self.readout(h)
        return y

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
        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)

        return self._forward_core(graph, w_adj)


class SparseRSGCN(RSGCN):
    """Sparse matrix version of RSGCN"""
    def __init__(self, out_dim, hidden_dim=32, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM,
                 use_batch_norm=False, readout=None):
        self._use_sparse_matrix = True
        super(SparseRSGCN, self).__init__(
            out_dim, hidden_dim, n_layers, n_atom_types, use_batch_norm,
            readout)

    def __call__(self, graph, adj_data, adj_row, adj_col):
        """Forward propagation

        Args:
            graph (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj_data (numpy.ndarray): minibatch of adjacency matrix.
            adj_row (numpy.ndarray): minibatch of adjacency matrix.
            adj_col (numpy.ndarray): minibatch of adjacency matrix.
                COO format is adopted as sparse matrix format. adj_data,
                adj_row and adj_col are data, row index and column index array
                of the matrix respectively.

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        adj_shape = [graph.shape[0], graph.shape[1], graph.shape[1]]

        if adj_data.ndim == 1:
            is_flatten = True
        elif adj_data.ndim == 2:
            is_flatten = False

        sp_adj = chainer_chemistry.functions.sparse_coo_matrix(
            adj_data, adj_row, adj_col, adj_shape, is_flatten=is_flatten)
        # adj_data/row/col: (minibatch, nnz)

        return self._forward_core(graph, sp_adj)
