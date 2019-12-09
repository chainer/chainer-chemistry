import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class RelGCNUpdate(chainer.Chain):
    """RelGUN submodule for update part.

    Args:
        in_channels (int or None): input channel dimension
        out_channels (int): output channel dimension
        num_edge_type (int): number of types of edge
    """

    def __init__(self, in_channels, out_channels, n_edge_types=4,
                 **kwargs):
        super(RelGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_self = GraphLinear(in_channels, out_channels)
            self.graph_linear_edge = GraphLinear(
                in_channels, out_channels * n_edge_types)
        self.n_edge_types = n_edge_types
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj, **kwargs):
        """main calculation

        Args:
            h: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns:
            (batchsize, num_nodes, ch)
        """
        mb, node, ch = h.shape

        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(h)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to
        # arbitrarily set it to 1
        m = self.graph_linear_edge(h)
        m = functions.reshape(
            m, (mb, node, self.out_channels, self.n_edge_types))
        m = functions.transpose(m, (0, 3, 1, 2))
        # m: (batchsize, edge_type, node, ch)
        # hrL (batchsize, edge_type, node, ch)
        hr = functions.matmul(adj, m)
        # hr: (batchsize, node, ch)
        hr = functions.sum(hr, axis=1)
        return hs + hr


class RelGCNSparseUpdate(chainer.Chain):
    """sparse RelGCN submodule for update part"""

    def __init__(self, in_channels, out_channels, n_edge_types):
        super(RelGCNSparseUpdate, self).__init__()
        self.out_channels = out_channels
        self.n_edge_types = n_edge_types
        with self.init_scope():
            self.root_weight = chainer.links.Linear(in_channels, out_channels)
            self.edge_weight = chainer.links.Linear(
                in_channels, n_edge_types * out_channels)

    def __call__(self, h, edge_index, edge_attr):
        next_h = self.root_weight(h)
        features = self.edge_weight(
            h) .reshape(-1, self.n_edge_types, self.out_channels)
        messages = features[edge_index[0], edge_attr, :]
        return functions.scatter_add(next_h, edge_index[1], messages)
