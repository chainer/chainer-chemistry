import chainer
import numpy
from chainer._backend import Device
from chainer_chemistry.dataset.graph_dataset.base_graph_data import \
    BaseGraphData
from chainer_chemistry.dataset.graph_dataset.feature_converters \
    import batch_with_padding, batch_without_padding, concat, shift_concat, \
    concat_with_padding, shift_concat_with_padding


class BaseGraphDataset(object):
    _pattern = ''
    _feature_entries = []
    _feature_batch_method = []

    def __init__(self, data_list, *args, **kwargs):
        self.data_list = data_list

    def register_feature(self, key, batch_method, skip_if_none=True):
        if skip_if_none and getattr(self.data_list[0], key, None) is None:
            return
        self._feature_entries.append(key)
        self._feature_batch_method.append(batch_method)

    def update_feature(self, key, batch_method):
        index = self._feature_entries.index(key)
        self._feature_batch_method[index] = batch_method

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]

    def converter(self, batch, device=None):
        if not isinstance(device, Device):
            device = chainer.get_device(device)
        batch = [method(name, batch, device=device) for name, method in
                 zip(self._feature_entries, self._feature_batch_method)]
        data = BaseGraphData(
            **{key: value for key, value in zip(self._feature_entries, batch)})
        return data


class PaddingGraphDataset(BaseGraphDataset):
    _pattern = 'padding'

    def __init__(self, data_list):
        super(PaddingGraphDataset, self).__init__(data_list)
        self.register_feature('x', batch_with_padding)
        self.register_feature('adj', batch_with_padding)
        self.register_feature('super_node', batch_with_padding)
        self.register_feature('pos', batch_with_padding)
        self.register_feature('y', batch_without_padding)
        self.register_feature('n_nodes', batch_without_padding)


class SparseGraphDataset(BaseGraphDataset):
    _pattern = 'sparse'

    def __init__(self, data_list):
        super(SparseGraphDataset, self).__init__(data_list)
        self.register_feature('x', concat)
        self.register_feature('edge_index', shift_concat)
        self.register_feature('edge_attr', concat)
        self.register_feature('super_node', concat)
        self.register_feature('pos', concat)
        self.register_feature('y', batch_without_padding)
        self.register_feature('n_nodes', batch_without_padding)

    def converter(self, batch, device=None):
        data = super(SparseGraphDataset, self).converter(batch, device=device)
        if not isinstance(device, Device):
            device = chainer.get_device(device)
        data.batch = numpy.concatenate([
            numpy.full((data.x.shape[0]), i, dtype=numpy.int)
            for i, data in enumerate(batch)
        ])
        data.batch = device.send(data.batch)
        return data

    def converter_with_padding(self, batch, device=None):
        self.update_feature('x', concat_with_padding)
        self.update_feature('edge_index', shift_concat_with_padding)
        data = super(SparseGraphDataset, self).converter(batch, device=device)
        if not isinstance(device, Device):
            device = chainer.get_device(device)
        max_n_nodes = max([data.x.shape[0] for data in batch])
        data.batch = numpy.concatenate([
            numpy.full((max_n_nodes), i, dtype=numpy.int)
            for i, data in enumerate(batch)
        ])
        data.batch = device.send(data.batch)
        return data
