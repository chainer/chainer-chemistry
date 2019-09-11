from chainer_chemistry.dataset.graph_dataset.base_graph_dataset import SparseGraphDataset  # NOQA
from chainer_chemistry.dataset.graph_dataset.base_graph_data import SparseGraphData  # NOQA
from chainer_chemistry.dataset.networkx_preprocessors.base_networkx import BaseSparseNetworkx  # NOQA


class GINSparseNetworkx(BaseSparseNetworkx):
    def __init__(self):
        pass

    def construct_sparse_data(self, graph):
        # self node feature is added in GINSParseUpdate
        # no need to add self loop
        return super(GINSparseNetworkx, self).construct_sparse_data(graph)
