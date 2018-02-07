"""
Implementation of Graph Convolutional Network
(https://arxiv.org/abs/1609.02907)
"""
import chainer
from chainer import cuda, Variable
from chainer import functions
import numpy

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM


class GCNUpdate(chainer.Chain):
    """GCN sub module for update part

    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
    """

    def __init__(self, in_channels, out_channels):
        super(GCNUpdate, self).__init__()
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


class GCNReadout(chainer.Chain):
    """GCN sub module for readout part

    Args:
        in_channels (int): dimension of feature vector associated to each node
        out_size (int): output dimension of feature vector associated to each graph
    """

    def __init__(self, in_channels, out_size):
        super(GCNReadout, self).__init__()
        with self.init_scope():
            self.output_weight = chainer_chemistry.links.GraphLinear(
                in_channels, out_size)
        self.in_channels = in_channels
        self.out_size = out_size

    def __call__(self, x):
        # --- Readout part ---
        h = self.output_weight(x)

        #
        h = functions.softmax(h, axis=2)  # softmax along channel axis
        y = functions.sum(h, axis=1)  # sum along node axis
        return y


class GCN(chainer.Chain):

    """Graph Convolutional Networks

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_atom_types (int): number of types of atoms
        n_layer (int): number of layers
    """

    def __init__(self, out_dim, hidden_dim=32, n_layers=4,
                 n_atom_types=MAX_ATOMIC_NUM):
        super(GCN, self).__init__()
        with self.init_scope():
            self.embed = chainer_chemistry.links.EmbedAtomID(
                in_size=n_atom_types, out_size=hidden_dim)
            self.gconvs = chainer.ChainList(
                *[GCNUpdate(hidden_dim, hidden_dim)
                  for _ in range(n_layers)])
            self.bnorms = chainer.ChainList(
                *[chainer_chemistry.links.GraphBatchNormalization(hidden_dim)
                  for _ in range(n_layers)])
            self.readout = GCNReadout(hidden_dim, out_dim)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

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

        # --- GCN update & readout ---
        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)
        
        for i, (gconv, bnorm) in enumerate(zip(self.gconvs,
                                               self.bnorms)):
            h = gconv(h, w_adj)
            h = bnorm(h)
            h = functions.dropout(h)
            if i < self.n_layers - 1:
                h = functions.relu(h)
            
        y = self.readout(h)
                
        return y
