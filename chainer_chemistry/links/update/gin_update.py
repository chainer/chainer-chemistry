import chainer
from chainer import functions

import chainer_chemistry
from chainer_chemistry.links import GraphMLP
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GINUpdate(chainer.Chain):
    r"""GIN submodule for update part.

    Simplest implementation of Graph Isomorphism Network (GIN):
    N-layered MLP + ReLU
    No learnable epsilon

    Batch Normalization is not implemented. instead we use dropout

    # TODO: implement Batch Normalization inside GraphMLP
    # Linear -> BN -> relu is used.

    See: Xu, Hu, Leskovec, and Jegelka, \
        "How powerful are graph neural networks?", in ICLR 2019.

    Args:
        in_channels (int or None): input dim of feature vector for each node
        hidden_channels (int): dimension of feature vector for each node
        out_channels (int or None): output dime of feature vector for each node
            When `None`, `hidden_channels` is used.
        dropout_ratio (float): ratio of dropout, instead of batch normalization
        n_layers (int): layers used in `GraphMLP`
    """

    def __init__(self, in_channels=None, hidden_channels=16, out_channels=None,
                 dropout_ratio=0.5, n_layers=2, **kwargs):
        if out_channels is None:
            out_channels = hidden_channels
        super(GINUpdate, self).__init__()
        channels = [hidden_channels] * (n_layers - 1) + [out_channels]
        with self.init_scope():
            # two Linear + RELU
            self.graph_mlp = GraphMLP(
                channels=channels, in_channels=in_channels,
                activation=functions.relu)
        self.dropout_ratio = dropout_ratio

    def __call__(self, h, adj, **kwargs):
        """Describing a layer.

        Args:
            h (numpy.ndarray): minibatch by num_nodes by hidden_dim
                numpy array. local node hidden states
            adj (numpy.ndarray): minibatch by num_nodes by num_nodes 1/0 array.
                Adjacency matrices over several bond types

        Returns:
            updated h
        """
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape

        # --- Message part ---
        # adj (mb, atom, atom)
        # fv   (minibatch, atom, ch)
        fv = chainer_chemistry.functions.matmul(adj, h)
        assert (fv.shape == (mb, atom, ch))

        # sum myself
        sum_h = fv + h
        assert (sum_h.shape == (mb, atom, ch))

        # apply MLP
        new_h = self.graph_mlp(sum_h)
        new_h = functions.relu(new_h)
        if self.dropout_ratio > 0.0:
            new_h = functions.dropout(new_h, ratio=self.dropout_ratio)
        return new_h
