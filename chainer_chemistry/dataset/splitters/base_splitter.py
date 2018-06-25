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

    def train_valid_test_split(self, dataset, frac_train=.8, frac_valid=.1,
                               frac_test=.1, seed=None, return_index=True,
                               converter=None):
        if converter is None:
            converter = converter_dict.get(type(dataset), converter_default)

        train_inds, valid_inds, test_inds = self._split(dataset,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=frac_test)
        if return_index:
            return train_inds, valid_inds, test_inds
        else:
            train = converter(dataset, train_inds)
            valid = converter(dataset, valid_inds)
            test = converter(dataset, test_inds)
            return train, valid, test,

    def train_valid_split(self, dataset, frac_train=.9, frac_valid=.1,
                          seed=None, return_index=True, converter=None):
        train_inds, valid_inds, test_inds = self._split(dataset,
                                                        frac_train=frac_train,
                                                        frac_valid=frac_valid,
                                                        frac_test=0)
        assert len(test_inds) == 0
        if converter is None:
            converter = converter_dict.get(type(dataset), converter_default)

        if return_index:
            return train_inds, valid_inds
        else:
            train = converter(dataset, train_inds)
            valid = converter(dataset, valid_inds)
            return train, valid,
