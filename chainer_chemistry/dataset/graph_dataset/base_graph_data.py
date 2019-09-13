import numpy
import chainer


class BaseGraphData(object):
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_device(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, (numpy.ndarray)):
                setattr(self, k, device.send(v))
            elif isinstance(v, (chainer.utils.CooMatrix)):
                data = device.send(v.data.array)
                row = device.send(v.row)
                col = device.send(v.col)
                gpu_coo_matrix = chainer.utils.CooMatrix(
                    data, row, col, v.shape, order=v.order)
                setattr(self, k, gpu_coo_matrix)
        return self


class PaddingGraphData(BaseGraphData):
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
