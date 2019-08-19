import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class RelGATUpdate(chainer.Chain):
    """RelGAT submodule for update part.

    Args:
        in_channels (int or None): dimension of input feature vector
        out_channels (int): dimension of output feature vector
        n_heads (int): number of multi-head-attentions.
        n_edge_types (int): number of edge types.
        dropout_ratio (float): dropout ratio of the normalized attention
            coefficients
        negative_slope (float): LeakyRELU angle of the negative slope
        softmax_mode (str): take the softmax over the logits 'across' or
            'within' relation. If you would like to know the detail discussion,
            please refer Relational GAT paper.
        concat_heads (bool) : Whether to concat or average multi-head
            attentions
    """
    def __init__(self, in_channels, out_channels, n_heads=3, n_edge_types=4,
                 dropout_ratio=-1., negative_slope=0.2, softmax_mode='across',
                 concat_heads=False):
        super(RelGATUpdate, self).__init__()
        with self.init_scope():
            self.message_layer = GraphLinear(
                in_channels, out_channels * n_edge_types * n_heads)
            self.attention_layer = GraphLinear(out_channels * 2, 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_edge_types = n_edge_types
        self.dropout_ratio = dropout_ratio
        self.softmax_mode = softmax_mode
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

    def __call__(self, h, adj, **kwargs):
        xp = self.xp
        # (minibatch, atom, channel)
        mb, atom, ch = h.shape
        # (minibatch, atom, EDGE_TYPE * heads * out_dim)
        h = self.message_layer(h)
        # (minibatch, atom, EDGE_TYPE, heads, out_dim)
        h = functions.reshape(h, (mb, atom, self.n_edge_types, self.n_heads,
                                  self.out_channels))
        # concat all pairs of atom
        # (minibatch, 1, atom, heads, out_dim)
        h_i = functions.reshape(h, (mb, 1, atom, self.n_edge_types,
                                    self.n_heads, self.out_channels))
        # (minibatch, atom, atom, heads, out_dim)
        h_i = functions.broadcast_to(h_i, (mb, atom, atom, self.n_edge_types,
                                           self.n_heads, self.out_channels))

        # (minibatch, atom, 1, EDGE_TYPE, heads, out_dim)
        h_j = functions.reshape(h, (mb, atom, 1, self.n_edge_types,
                                    self.n_heads, self.out_channels))
        # (minibatch, atom, atom, EDGE_TYPE, heads, out_dim)
        h_j = functions.broadcast_to(h_j, (mb, atom, atom, self.n_edge_types,
                                           self.n_heads, self.out_channels))

        # (minibatch, atom, atom, EDGE_TYPE, heads, out_dim * 2)
        e = functions.concat([h_i, h_j], axis=5)

        # (minibatch, EDGE_TYPE, heads, atom, atom, out_dim * 2)
        e = functions.transpose(e, (0, 3, 4, 1, 2, 5))
        # (minibatch * EDGE_TYPE * heads, atom * atom, out_dim * 2)
        e = functions.reshape(e, (mb * self.n_edge_types * self.n_heads,
                                  atom * atom, self.out_channels * 2))
        # (minibatch * EDGE_TYPE * heads, atom * atom, 1)
        e = self.attention_layer(e)

        # (minibatch, EDGE_TYPE, heads, atom, atom)
        e = functions.reshape(e, (mb, self.n_edge_types, self.n_heads, atom,
                                  atom))
        e = functions.leaky_relu(e, self.negative_slope)

        # (minibatch, EDGE_TYPE, atom, atom)
        if isinstance(adj, chainer.Variable):
            cond = adj.array.astype(xp.bool)
        else:
            cond = adj.astype(xp.bool)
        # (minibatch, EDGE_TYPE, 1, atom, atom)
        cond = xp.reshape(cond, (mb, self.n_edge_types, 1, atom, atom))
        # (minibatch, EDGE_TYPE, heads, atom, atom)
        cond = xp.broadcast_to(cond, e.array.shape)
        # TODO(mottodora): find better way to ignore non connected
        e = functions.where(cond, e,
                            xp.broadcast_to(xp.array(-10000), e.array.shape)
                            .astype(xp.float32))
        # In Relational Graph Attention Networks eq.(7)
        # ARGAT: take the softmax over the logits across node neighborhoods
        # irrespective of relation
        if self.softmax_mode == 'across':
            # (minibatch, heads, atom, EDGE_TYPE, atom)
            e = functions.transpose(e, (0, 2, 3, 1, 4))
            # (minibatch, heads, atom, EDGE_TYPE * atom)
            e = functions.reshape(e, (mb, self.n_heads, atom,
                                      self.n_edge_types * atom))
            # (minibatch, heads, atom, EDGE_TYPE * atom)
            alpha = functions.softmax(e, axis=3)
            if self.dropout_ratio >= 0:
                alpha = functions.dropout(alpha, ratio=self.dropout_ratio)
            # (minibatch, heads, atom, EDGE_TYPE, atom)
            alpha = functions.reshape(alpha, (mb, self.n_heads, atom,
                                              self.n_edge_types, atom))
            # (minibatch, EDGE_TYPE, heads, atom, atom)
            alpha = functions.transpose(alpha, (0, 3, 1, 2, 4))

        # In Relational Graph Attention Networks eq.(6)
        # WIRGAT: take the softmax over the logits independently for each
        # relation
        elif self.softmax_mode == 'within':
            alpha = functions.softmax(e, axis=4)
            if self.dropout_ratio >= 0:
                alpha = functions.dropout(alpha, ratio=self.dropout_ratio)
        else:
            raise ValueError("{} is invalid. Please use 'across' or 'within'"
                             .format(self.softmax_mode))

        # before: (minibatch, atom, EDGE_TYPE, heads, out_dim)
        # after: (minibatch, EDGE_TYPE, heads, atom, out_dim)
        h = functions.transpose(h, (0, 2, 3, 1, 4))
        # (minibatch, EDGE_TYPE, heads, atom, out_dim)
        h_new = functions.matmul(alpha, h)
        # (minibatch, heads, atom, out_dim)
        h_new = functions.sum(h_new, axis=1)
        if self.concat_heads:
            # -> (minibatch, atom, heads, out_dim)
            h_new = functions.transpose(h_new, (0, 2, 1, 3))
            bs, n_nodes, n_heads, outdim = h_new.shape
            # (minibatch, atom, heads * out_dim)
            h_new = functions.reshape(h_new, (bs, n_nodes, n_heads * outdim))
        else:
            # (minibatch, atom, out_dim)
            h_new = functions.mean(h_new, axis=1)
        return h_new
