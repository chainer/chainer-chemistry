import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class RelGCNUpdate(chainer.Chain):
    """RelGUN submodule for update part.

    Args:
        in_channels (int): input channel dimension
        out_channels (int): output channel dimension
        num_edge_type (int): number of types of edge
    """

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        super(RelGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_self = GraphLinear(in_channels, out_channels)
            self.graph_linear_edge = GraphLinear(
                in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj):
        """

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
            m, (mb, node, self.out_channels, self.num_edge_type))
        m = functions.transpose(m, (0, 3, 1, 2))
        # m: (batchsize, edge_type, node, ch)
        # hrL (batchsize, edge_type, node, ch)
        hr = functions.matmul(adj, m)
        # hr: (batchsize, node, ch)
        hr = functions.sum(hr, axis=1)
        return hs + hr
