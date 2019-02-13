import chainer
from chainer import functions as F
from chainer import links as L

import chainer_chemistry
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GINUpdate(chainer.Chain):
    """GIN submodule for update part.

    Simplest implementation of Graph Isomorphism Network (GIN):
    2-layered MLP + ReLU
    no learnble epsilon

    Batch Normalization is not implemetned. instead we use droout

    # TODO: implement Batch Normalization
    # TODO: use GraphMLP instead of GraphLinears

    See: Xu, Hu, Leskovec, and Jegelka, "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        hidden_dim (int): dimension of feature vector associated to
            each atom
        dropout_ratio (float): ratio of dropout, insted of bach normlization
    """

    def __init__(self, hidden_dim=16, dropout_ratio=0.5):
        super(GINUpdate, self).__init__()
        with self.init_scope():
            # two Linear + RELU
            self.linear_g1 = GraphLinear(hidden_dim, hidden_dim)
            self.linear_g2 = GraphLinear(hidden_dim, hidden_dim)
        # end with
        self.dropout_ratio = dropout_ratio
    # end-def

    def __call__(self, h, adj):
        """
        Describing a layer.

        Args:
            h (numpy.ndarray): minibatch by num_nodes by hidden_dim numpy array.
                local node hidden states
            adj (numpy.ndarray): minibatch by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types

        Returns:
            updated h

        """

        xp = self.xp

        # (minibatch, atom, ch)
        mb, atom, ch = h.shape

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
        new_h = F.relu(self.linear_g1(sum_h))
        if self.dropout_ratio > 0.0:
            new_h = F.relu(F.dropout(self.linear_g2(new_h),ratio=self.dropout_ratio))
        else:
            new_h = F.relu(self.linear_g2(new_h))

        # done???
        return new_h

    def reset_state(self):
        self.update_layer.reset_state()
