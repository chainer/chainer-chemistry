import chainer
from chainer import functions
import numpy

import chainer_chemistry
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class NFPUpdate(chainer.Chain):
    """NFP submodule for update part.

    Args:
        in_channels (int or None): input channel dimension
        out_channels (int): output channel dimension
        max_degree (int): max degree of edge
    """

    def __init__(self, in_channels, out_channels, max_degree=6,
                 **kwargs):
        super(NFPUpdate, self).__init__()
        num_degree_type = max_degree + 1
        with self.init_scope():
            self.graph_linears = chainer.ChainList(
                *[GraphLinear(in_channels, out_channels)
                  for _ in range(num_degree_type)])
        self.max_degree = max_degree
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj, deg_conds):
        # h: (minibatch, atom, ch)
        # h encodes each atom's info in ch axis of size hidden_dim
        # adjs: (minibatch, atom, atom)

        # --- Message part ---
        # Take sum along adjacent atoms

        # fv: (minibatch, atom, ch)
        fv = chainer_chemistry.functions.matmul(adj, h)

        # --- Update part ---
        # TODO(nakago): self.xp is chainerx
        if self.xp is numpy:
            zero_array = numpy.zeros(fv.shape, dtype=numpy.float32)
        else:
            zero_array = self.xp.zeros_like(fv.array)

        fvds = [functions.where(cond, fv, zero_array) for cond in deg_conds]

        out_h = 0
        for graph_linear, fvd in zip(self.graph_linears, fvds):
            out_h = out_h + graph_linear(fvd)

        # out_h shape (minibatch, max_num_atoms, hidden_dim)
        out_h = functions.sigmoid(out_h)
        return out_h
