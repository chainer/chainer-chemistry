import chainer
from chainer._backend import Device
from chainer_chemistry.dataset.graph_dataset.base_graph_data import \
    BaseGraphData
from chainer_chemistry.dataset.graph_dataset.feature_converters import padding, \
    concat, shift_concat


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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]

    def converter(self, batch, device=None):
        if not isinstance(device, Device):
            device = chainer.get_device(device)
        batch = [method(name, batch, device=device) for name, method in
                 zip(self._feature_entries, self._feature_batch_method)]
        data = BaseGraphData(**{key: value for key, value in zip(self._feature_entries, batch)})
        return data


class PaddingGraphDataset(BaseGraphDataset):
    _pattern = 'padding'

    def __init__(self, data_list):
        super(PaddingGraphDataset, self).__init__(data_list)
        self.register_feature('x', padding)
        self.register_feature('adj', padding)
        self.register_feature('super_node', padding)
        self.register_feature('pos', padding)
        self.register_feature('y', padding)


class SparseGraphDataset(BaseGraphDataset):
    _pattern = 'sparse'

    def __init__(self, data_list):
        super(SparseGraphDataset, self).__init__(data_list)
        self.register_feature('x', concat)
        self.register_feature('edge_index', shift_concat)
        self.register_feature('edge_attr', shift_concat)
        self.register_feature('super_node', concat)
        self.register_feature('pos', padding)
        self.register_feature('y', padding)
