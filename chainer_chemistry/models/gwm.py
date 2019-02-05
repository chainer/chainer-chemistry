# " -*- coding: utf-8 -*-"
# ----------------------------------------------------------------------
# Name:        gwm.py
# Purpose:     Implementation of the Graph Warp Module with a single super node
#
#              inputs:
#
#              outputs:
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     29/10/18 (DD/MM/YY)
# Last update: 05/02/19 (DD/MM/YY)
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

from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


class GWM(chainer.Chain):
    """
    Graph Warping Module (GWM)

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module for Boosting the Power of Graph Neural Networks", arXiv, 2019.

    Args:
        hidden_dim (default=16): dimension of hidden vectors
            associated to each atom (local node)
        hiden_dim_super(default=16); dimension of super-node hidden vector
        n_layers (default=4): number of layers
        n_heads (default=8): numbef of heads
        n_atom_types (default=MAX_ATOMIC_NUM): number of types of atoms
        n_super_feature (default: tuned according to gtn_preprocessor); number of super-node observation attributes
        n_edge_types (int): number of edge types witin graphs.
        dropout_ratio (default=0.5); if > 0.0, perform dropout
        tying_flag (default=false): enable if you want to share params across layers
    """
    NUM_EDGE_TYPE = 4

    def __init__(self, hidden_dim=16, hidden_dim_super=16,
                 n_layers=4, n_heads=8,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 tying_flag=False,
                 gpu=-1):
        super(GWM, self).__init__()
        num_layer = n_layers
        if tying_flag:
            num_layer = 1

        with self.init_scope():

            #
            # for super-node unit
            #

            self.F_super = chainer.ChainList(
                *[L.Linear(in_size=hidden_dim_super, out_size=hidden_dim_super)
                  for _ in range(num_layer)]
            )

            #
            # for Transmitter unit
            #

            self.V_super = chainer.ChainList(
                *[L.Linear(hidden_dim * n_heads, hidden_dim * n_heads)
                  for _ in range(num_layer)]
            )
            self.W_super = chainer.ChainList(
                *[L.Linear(hidden_dim * n_heads, hidden_dim_super)
                  for _ in range(num_layer)]
            )
            self.B = chainer.ChainList(
                *[GraphLinear(n_heads * hidden_dim, n_heads * hidden_dim_super)
                  for _ in range(num_layer)]
            )

            #
            # for Merger Gate unit
            #
            self.gate_dim = hidden_dim
            self.H_local = chainer.ChainList(
                *[GraphLinear(in_size=hidden_dim, out_size=self.gate_dim)
                  for _ in range(num_layer)]
            )
            self.G_local = chainer.ChainList(
                *[GraphLinear(in_size=hidden_dim_super, out_size=self.gate_dim)
                  for _ in range(num_layer)]
            )

            self.gate_dim_super = hidden_dim_super
            self.H_super = chainer.ChainList(
                *[L.Linear(in_size=hidden_dim, out_size=self.gate_dim_super)
                  for _ in range(num_layer)]
            )
            self.G_super = chainer.ChainList(
                *[L.Linear(in_size=hidden_dim_super, out_size=self.gate_dim_super)
                  for _ in range(num_layer)]
            )

            # GRU's. not layer-wise (recurrent through layers)

            self.GRU_local = L.GRU(in_size=hidden_dim, out_size=hidden_dim)
            self.GRU_super = L.GRU(in_size=hidden_dim_super, out_size=hidden_dim_super)
        # end init_scope-with

        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.tying_flag = tying_flag

    def __call__(self, h, h_new, g, step=0):
        """
        Describes the module for a single layer update.
        Do not forget to rest GRU for each batch...

        :param h: minibatch by num_nodes by hidden_dim numpy array.
                current local node hidden states as input of the vanilla GNN
        :param h_new: minibatch by num_nodes by hidden_dim numpy array.
                updated local node hidden states as output from the vanilla GNN
        :param adj: minibatch by bond_types by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types
        :param g: minibatch by hidden_dim_super numpy array.
                current super node hiddden state
        :param step: integer, the layer index
        :return: updated h and g
        """

        xp = self.xp

        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        out_ch = ch

        #
        # Super node unit: non linear update of the super node
        #

        g_new = F.relu(self.F_super[step](g))


        #
        # Transmodule unit: inter-module message passing
        #

        #
        # h_trans: local --> super transmission
        #

        h1 = F.expand_dims(h, 2)
        #assert h1.shape == (mb, atom, 1, ch)
        h1 = F.broadcast_to(h1, [mb, atom, self.n_heads, ch])
        h1 = F.reshape(h1, [mb, atom, self.n_heads* ch])
        #assert h1.shape==(mb, atom, self.n_heads * ch)
        h_j = F.expand_dims(h, 1)
        h_j = F.broadcast_to(h_j, (mb, self.n_heads, atom, ch))
        #assert h_j.shape==(mb, self.n_heads, atom, ch)

        # expand h_super
        g_extend = F.expand_dims(g, 1)
        # assert g_extend.shape==(mb, 1, self.hidden_dim_super)
        g_extend = F.broadcast_to(g_extend, (mb, self.n_heads, self.hidden_dim_super))
        # assert g_extend.shape==(mb, self.n_heads, self.hidden_dim_super)
        g_extend = F.expand_dims(g_extend, 2)
        # assert g_extend.shape==(mb, self.n_heads, 1, self.hidden_dim_super)

        # update for attention-message B h_i
        # mb, atom, n_heads * ch
        Bh_i = self.B[step](h1)
        # assert Bh_i.shape==(mb, atom, self.n_heads * self.hidden_dim_super)
        # mb, atom, num_head, ch
        Bh_i = F.reshape(Bh_i, [mb, atom, self.n_heads, self.hidden_dim_super])
        # mb, num_head, atom, ch
        Bh_i = F.transpose(Bh_i, [0, 2, 1, 3])
        # assert Bh_i.shape==(mb, self.n_heads, atom, self.hidden_dim_super)

        # take g^{T} * B * h_i
        # indexed by i
        # mb, self.n_haeds atom(i)
        b_hi = F.matmul(g_extend, Bh_i, transb=True)  # This will reduce the last hidden_dim_super axis
        # assert b_hi.shape==(mb, self.n_heads, 1, atom)

        # softmax. sum/normalize over the last axis.
        # mb, self.n_heda, atom(i-normzlied)
        attention_i = F.softmax(b_hi, axis=3)
        if self.dropout_ratio > 0.0:
            attention_i = F.dropout(attention_i,ratio=self.dropout_ratio)
        # assert attention_i.shape==(mb, self.n_heads, 1, atom)

        # element-wise product --> sum over i
        # mb, num_head, hidden_dim_super
        attention_sum = F.matmul(attention_i, h_j)
        # assert attention_sum.shape==(mb, self.n_heads, 1, ch)
        attention_sum = F.reshape(attention_sum, (mb, self.n_heads * ch))
        # assert attention_sum.shape==(mb, self.n_heads * ch)

        # weighting h for different heads
        h_trans = self.V_super[step](attention_sum)
        # assert intermediate_h.shape==(mb, self.n_heads * ch)
        # compress heads
        h_trans = self.W_super[step](h_trans)
        h_trans = F.tanh(h_trans)
        # assert intermediate_h.shape==(mb, self.hidden_dim_super)


        #
        # g_trans: super --> local transmission
        #

        # for local updates
        g_trans = self.F_super[step](g)
        g_trans = F.tanh(g_trans)
        # assert intermediate_h_super.shape==(mb, self.hidden_dim)
        g_trans = F.expand_dims(g_trans, 1)
        # assert intermediate_h_super.shape==(mb, 1, self.hidden_dim)
        g_trans = F.broadcast_to(g_trans, (mb, atom, self.hidden_dim))
        # assert intermediate_h_super.shape==(mb, atom, self.hidden_dim)


        #
        # Gated Merger unit
        #
        z_local = self.H_local[step](h_new) + self.G_local[step](g_trans)
        z_local = F.broadcast_to(z_local, (mb, atom, self.hidden_dim))
        if self.dropout_ratio > 0.0:
            z_local = F.dropout(z_local,ratio=self.dropout_ratio)
        z_local = F.sigmoid(z_local)
        merged_h = (1.0-z_local) * h_new + z_local * g_trans
        # assert new_h.shape==(mb, atom, ch)

        z_super = self.H_super[step](h_trans) + self.G_super[step](g_new)
        z_super = F.broadcast_to(z_super, (mb, self.hidden_dim_super))
        if self.dropout_ratio > 0.0:
            z_super = F.dropout(z_super,ratio=self.dropout_ratio)
        z_super = F.sigmoid(z_super)
        merged_g = (1.0-z_super) * h_trans + z_super * g_new
        # assert out_h_super.shape==(mb, self.hidden_dim_super)

        #
        # --- feed to GRU for final self-gating ---
        #
        out_h = F.reshape(merged_h, (mb * atom, self.hidden_dim))
        out_h = self.GRU_local(out_h)
        out_h = F.reshape(out_h, (mb, atom, self.hidden_dim))

        out_g = self.GRU_super(merged_g)

        return out_h, out_g
