# " -*- coding: utf-8 -*-"
# ----------------------------------------------------------------------
# Name:        gin.py
# Purpose:     Implementation of the simplest case of Graph Isomorphism Network( GIN)
#              no learnable epsilon
#              2-layer MLP + ReLU
#
#              inputs:
#
#              outputs:
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     13/12/18 (DD/MM/YY)
# Last update: 13/12/18 (DD/MM/YY)
# -----------------------------------------------------------------------

import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


class GIN(chainer.Chain):
    """
    Simplest implementation of Graph Isomorphism Network (GIN)

    See: Ishiguro, Maeda, and Koyama To Be Written.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (default=16): dimension of hidden vectors
            associated to each atom
        hiden_dim_super(default=16); dimension of super-node hidden vector
        n_layers (default=4): number of layers
        n_heads (default=8): numbef of heads
        n_atom_types (default=MAX_ATOMIC_NUM): number of types of atoms
        n_super_feature (default: tuned according to gtn_preprocessor); number of super-node observation attributes
        n_edge_types (int): number of edge types witin graphs.
        dropout_ratio (default=0.5); if > 0.0, perform dropout
        concat_hidden (default=False): If set to True, readout is executed in each layer
            and the result is concatenated
        tying_flag (default=True): enable weight_tying for all units
        scaler_mgr_flag (default=False): reduce the merger gate to be scalar.

    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 weight_tying=True,
                 gpu=-1):
        super(GIN, self).__init__()

        num_layer = 1 if weight_tying else n_layers
        n_readout_layer = n_layers if concat_hidden else 1
        with self.init_scope():
            # embedding
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)

            # two non-linear MLP part
            self.linear_g1 = chainer.ChainList(
                *[GraphLinear(hidden_dim, hidden_dim)
                for _ in range(num_layer)]
            )

            self.linear_g2 = chainer.ChainList(
                *[GraphLinear(hidden_dim, hidden_dim)
                for _ in range(num_layer)]
            )

            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )

        # end init_scope-with
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def update(self, h, adj, step=0):
        """
        Describes the each layer.

        :param h: minibatch by num_nodes by hidden_dim numpy array.
                local node hidden states
        :param adj: minibatch by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types
        :param step: integer, the layer index
        :return: updated h and h_super
        """

        xp = self.xp

        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch

        layer_index = 0 if self.weight_tying else step

        # --- Message part ---
        # Take sum along adjacent atoms

        # adj (mb, atom, atom)
        # fv   (minibatch, atom, ch)
        fv = chainer_chemistry.functions.matmul(adj, h)
        assert(fv.shape == (mb, atom, ch) )

        # sum myself
        sum_h = fv + h
        assert(sum_h.shape == (mb, atom, ch))

        # apply MLP
        new_h = F.relu(self.linear_g1[layer_index](sum_h))
        new_h = F.relu(F.dropout(self.linear_g2[layer_index](new_h),ratio=self.dropout_ratio))

        # done???

        return new_h

    def readout(self, h, h0, step=0):
        """
        Readout (aggregation) throuth tow MLPs.

        :param h:  minibatch by num_nodes by hidden_dim numppy.ndarray, output from the Conv layers
        :param h0: minibatch by num_nodes by feature_dim (= num_max_atom) numpy.ndarray, input local node features
        :param step: integer, index for layers.
        :return:
        """
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        # h, h0: (minibatch, atom, ch)
        g = F.sigmoid(
            self.i_layers[index](F.concat((h, h0), axis=2))) \
            * self.j_layers[index](h)
        g = F.sum(g, axis=1)  # sum along atom's axis
        return g

    def __call__(self, atom_array, adj):
        """
        Forward propagation

        :param atom_array - mol-minibatch by node numpy.ndarray,
                minibatch of molecular which is represented with atom IDs (representing C, O, S, ...)
                atom_array[m, i] = a represents
                m-th molecule's i-th node is value a (atomic number)
        :param adj  - mol-minibatch by relation-types by node by node numpy.ndarray,
                       minibatch of multple relational adjancency matrix with edge-type information
                       adj[m, i, j] = b represents
                       m-th molecule's  edge from node i to node j has value b
        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """

        # assert len(super_node) > 0
        # print("for DEBUG: graphtransformer.py::__call__(): len(super_node)=" + str(len(super_node)))

        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        # end if-else
        # print("for DEBUG: graphtransformer.py::__call__(): xp.shape(h)=" + str(xp.shape(h)))


        h0 = F.copy(h, cuda.get_device_from_array(h.data).id)

        g_list = []
        for step in range(self.n_layers):
            h = self.update(h, adj, step)
            if self.concat_hidden:
                g = self.readout(h, h0, step)
                g_list.append(g)

        if self.concat_hidden:
            return F.concat(g_list, axis=1)
        else:
            g1 = self.readout(h, h0, 0)
            return g1
