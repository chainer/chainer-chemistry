import numpy
from chainer_chemistry.dataset.graph_dataset.base_graph_dataset import SparseGraphDataset  # NOQA
from chainer_chemistry.dataset.graph_dataset.base_graph_data import SparseGraphData  # NOQA
from chainer_chemistry.dataset.graph_dataset.feature_converters import batch_without_padding  # NOQA


class BaseSparseNetworkx():
    """
    Preprocess NetworkX::Graph for each model's input
    """
    def __init__(self):
        pass

    def construct_sparse_data(self, graph):
        edge_index = [[], []]
        for e in graph.edges:
            edge_index[0].append(e[0])
            edge_index[1].append(e[1])

        # x
        if 'x' in graph.graph:
            x = graph.graph['x']
        else:
            feature_dim, = graph.nodes[0]['x'].shape
            x = numpy.empty((graph.number_of_nodes(), feature_dim),
                            dtype=numpy.float32)
            for v, data in graph.nodes.data():
                x[v] = data['x']

        # y
        if 'y' in graph.graph:
            y = graph.graph['y']
        else:
            y = numpy.empty(graph.number_of_nodes(), dtype=numpy.int32)
            for v, data in graph.nodes.data():
                y[v] = data['y']

        return SparseGraphData(
            x=x,
            edge_index=numpy.array(edge_index, dtype=numpy.int),
            y=y,
            label_num=graph.graph['label_num']
        )

    def add_self_loop(self, graph):
        for v in range(graph.number_of_nodes()):
            graph.add_edge(v, v)
        return graph

    def create_dataset(self, graph_list):
        data_list = [
            self.construct_sparse_data(graph) for graph in graph_list
        ]
        dataset = SparseGraphDataset(data_list)
        dataset.register_feature('label_num', batch_without_padding)
        return dataset
