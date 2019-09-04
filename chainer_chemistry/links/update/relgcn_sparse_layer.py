import chainer
from chainer import functions


class RelGCNSparseLayer(chainer.Chain):
    def __init__(self, in_channels, out_channels, n_edge_types):
        super(RelGCNSparseLayer, self).__init__()
        self.out_channels = out_channels
        self.n_edge_types = n_edge_types
        with self.init_scope():
            self.root_weight = chainer.links.Linear(in_channels, out_channels)
            self.edge_weight = chainer.links.Linear(
                in_channels, n_edge_types * out_channels)

    def __call__(self, h, graph):
        next_h = self.root_weight(h)
        features = self.edge_weight(
            h) .reshape(-1, self.n_edge_types, self.out_channels)
        messages = features[graph.edge_index[0], graph.edge_attr, :]
        return functions.scatter_add(next_h, graph.edge_index[1], messages)
