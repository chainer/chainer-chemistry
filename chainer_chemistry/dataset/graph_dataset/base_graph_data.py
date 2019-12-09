import numpy

import chainer


class BaseGraphData(object):
    """Base class of graph data """

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_device(self, device):
        """Send self to `device`

        Args:
            device (chainer.backend.Device): device

        Returns:
            self sent to `device`
        """
        for k, v in self.__dict__.items():
            if isinstance(v, (numpy.ndarray)):
                setattr(self, k, device.send(v))
            elif isinstance(v, (chainer.utils.CooMatrix)):
                data = device.send(v.data.array)
                row = device.send(v.row)
                col = device.send(v.col)
                device_coo_matrix = chainer.utils.CooMatrix(
                    data, row, col, v.shape, order=v.order)
                setattr(self, k, device_coo_matrix)
        return self


class PaddingGraphData(BaseGraphData):
    """Graph data class for padding pattern

    Args:
        x (numpy.ndarray): input node feature
        adj (numpy.ndarray): adjacency matrix
        y (int or numpy.ndarray): graph or node label
    """

    def __init__(self, x=None, adj=None, super_node=None, pos=None, y=None,
                 **kwargs):
        self.x = x
        self.adj = adj
        self.super_node = super_node
        self.pos = pos
        self.y = y
        self.n_nodes = x.shape[0]
        super(PaddingGraphData, self).__init__(**kwargs)


class SparseGraphData(BaseGraphData):
    """Graph data class for sparse pattern

    Args:
        x (numpy.ndarray): input node feature
        edge_index (numpy.ndarray): sources and destinations of edges
        edge_attr (numpy.ndarray): attribution of edges
        y (int or numpy.ndarray): graph or node label
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 pos=None, super_node=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.super_node = super_node
        self.y = y
        self.n_nodes = x.shape[0]
        super(SparseGraphData, self).__init__(**kwargs)
