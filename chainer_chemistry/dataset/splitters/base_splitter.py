from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


def converter_default(dataset, indices):
    return dataset[indices]


def converter_numpy_tuple_dataset(dataset, indices):
    return NumpyTupleDataset(*dataset.features[indices])


converter_dict = {
    NumpyTupleDataset: converter_numpy_tuple_dataset
}


class BaseSplitter(object):
    def k_fold_split(self, dataset, k):
        raise NotImplementedError

    def _split(self, dataset):
        raise NotImplementedError

    def train_valid_test_split(self, dataset, **kwargs):
        converter = kwargs.get('converter', None)
        return_index = kwargs.get('return_index', True)
        kwargs.setdefault('frac_train', 0.8)
        kwargs.setdefault('frac_valid', 0.1)
        kwargs.setdefault('frac_test', 0.1)

        if converter is None:
            converter = converter_dict.get(type(dataset), converter_default)

        train_inds, valid_inds, test_inds = self._split(dataset, **kwargs)

        if return_index:
            return train_inds, valid_inds, test_inds
        else:
            train = converter(dataset, train_inds)
            valid = converter(dataset, valid_inds)
            test = converter(dataset, test_inds)
            return train, valid, test,

    def train_valid_split(self, dataset, **kwargs):
        converter = kwargs.get('converter', None)
        return_index = kwargs.get('return_index', True)
        kwargs.setdefault('frac_train', 0.9)
        kwargs.setdefault('frac_valid', 0.1)
        kwargs.setdefault('frac_test', 0.0)

        train_inds, valid_inds, test_inds = self._split(dataset, **kwargs)
        assert len(test_inds) == 0

        if converter is None:
            converter = converter_dict.get(type(dataset), converter_default)

        if return_index:
            return train_inds, valid_inds
        else:
            train = converter(dataset, train_inds)
            valid = converter(dataset, valid_inds)
            return train, valid,
