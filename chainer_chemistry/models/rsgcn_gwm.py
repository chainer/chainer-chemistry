"""
Implementation of Renormalized Spectral Graph Convolutional Network (RSGCN)
w/ Graph Warping Module (GWM)

See: Thomas N. Kipf and Max Welling, \
    Semi-Supervised Classification with Graph Convolutional Networks. \
    September 2016. \
    `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_
"""
import chainer
from chainer import functions
from chainer import Variable
import chainer.links as L
import chainer.functions as F

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM

from chainer_chemistry.models import gwm
from chainer_chemistry.models.gwm import GWM


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


class RSGCN_GWM(chainer.Chain):

    """Renormalized Spectral Graph Convolutional Network (RSGCN)
    with Graph Warp Module (GWM)

    See: Thomas N. Kipf and Max Welling, \
        Semi-Supervised Classification with Graph Convolutional Networks. \
        September 2016. \
        `arXiv:1609.02907 <https://arxiv.org/abs/1609.02907>`_

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module for Boosting the Power of Graph Neural Networks", arXiv, 2019.

    The name of this model "Renormalized Spectral Graph Convolutional Network
    (RSGCN)" is named by us rather than the authors of the paper above.
    The authors call this model just "Graph Convolution Network (GCN)", but
    we think that "GCN" is bit too general and may cause namespace issue.
    That is why we did not name this model as GCN.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        hiden_dim_super(default=16); dimension of super-node hidden vector
        n_layers (int): number of layers
        n_heads (default=8): numbef of heads
        n_atom_types (int): number of types of atoms
        n_super_feature (default: tuned according to gtn_preprocessor); number of super-node observation attributes
        use_batch_norm (bool): If True, batch normalization is applied after
            graph convolution.
        readout (Callable): readout function. If None, `rsgcn_readout_sum` is
            used. To the best of our knowledge, the paper of RSGCN model does
            not give any suggestion on readout.
        dropout_ratio (float): ratio used in dropout function.
            If 0 or negative value is set, dropout function is skipped.
        weight_tying (default=True): enable weight_tying for all units
        scaler_mgr_flag (default=False): reduce the merger gate to be scalar.

    """

    def __init__(self, out_dim, hidden_dim=32, hidden_dim_super=16, n_layers=4, n_heads=8,
                 n_atom_types=MAX_ATOMIC_NUM,
                 n_super_feature=4 + 2 + 4 + MAX_ATOMIC_NUM*2,
                 use_batch_norm=False, readout=None, dropout_ratio=0.5,
                 weight_tying=False,
                 scaler_mgr_flag=False,):
        super(RSGCN_GWM, self).__init__()
        in_dims = [hidden_dim for _ in range(n_layers)]
        out_dims = [hidden_dim for _ in range(n_layers)]
        out_dims[n_layers - 1] = out_dim
        with self.init_scope():
            self.embed = chainer_chemistry.links.EmbedAtomID(
                in_size=n_atom_types, out_size=hidden_dim)
            self.embed_super = L.Linear(in_size=n_super_feature, out_size=hidden_dim_super)

            self.gconvs = chainer.ChainList(
                *[RSGCNUpdate(in_dims[i], out_dims[i])
                  for i in range(n_layers)])
            if use_batch_norm:
                self.bnorms = chainer.ChainList(
                    *[chainer_chemistry.links.GraphBatchNormalization(
                        out_dims[i]) for i in range(n_layers)])
            else:
                self.bnorms = [None for _ in range(n_layers)]

            # GWM
            self.gwm = GWM(hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                           n_layers=n_layers, n_heads=n_heads,
                           dropout_ratio=dropout_ratio,
                           tying_flag=weight_tying,
                           scaler_mgr_flag=scaler_mgr_flag,
                           gpu=-1)

            if isinstance(readout, chainer.Link):
                self.readout = readout

            self.linear_for_concat_super = L.Linear(in_size=None, out_size=out_dim)
        if not isinstance(readout, chainer.Link):
            self.readout = readout or rsgcn_readout_sum
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.weight_tying = weight_tying

    def update(self, h, adj, g, step=0):
        """
        Describes the each layer.

        :param h: minibatch by num_nodes by hidden_dim numpy array.
                local node hidden states
        :param adj: minibatch by bond_types by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types
        :param g: minibatch by hidden_dim_super numpy array.
                super node hidddne state
        :param bnorm: batch normalizer
        :param step: integer, the layer index
        :return: updated h and h_super
        """

        xp = self.xp

        out_h = self.gconvs[step](h, adj)
        if self.bnorms[step] is not None:
            out_h = self.bnorms[step](out_h)
            if self.dropout_ratio > 0.:
                out_h = functions.dropout(out_h, ratio=self.dropout_ratio)
            if step < self.n_layers - 1:
                out_h = functions.relu(out_h)

        #
        # Graph Warping Module
        #
        step_gwm = step
        if self.weight_tying:
            step_gwm = 0
        new_h, new_g = self.gwm(h, out_h, g, step_gwm)

        return new_h, new_g

    def __call__(self, graph, adj, super_node):
        """Forward propagation

        Args:
            graph (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix
                `adj[mol_index]` represents `mol_index`-th molecule's
                adjacency matrix
        :param super_node  - 1D numpy.ndarray, the super-node observation.

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

        # call reset for all RNN modules in GWM
        self.gwm.GRU_local.reset_state()
        self.gwm.GRU_super.reset_state()
        # ebmbed super node
        g = self.embed_super(super_node)

        # --- RSGCN update ---
        for i in range(len(self.gconvs)):
            h, g = self.update(h, w_adj, g, i)

        #for i, (gconv, bnorm) in enumerate(zip(self.gconvs,
        #                                       self.bnorms)):
            #h = gconv(h, w_adj)
            #if bnorm is not None:
            #    h = bnorm(h)
            #if self.dropout_ratio > 0.:
            #    h = functions.dropout(h, ratio=self.dropout_ratio)
            #if i < self.n_layers - 1:
            #    h = functions.relu(h)

        # --- readout ---
        y = self.readout(h)

        g2 = F.concat((y, g))
        out_g = F.relu(self.linear_for_concat_super(g2))

        return out_g
