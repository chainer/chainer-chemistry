import numpy
import pandas

from chainer_chemistry.dataset.splitters.base_splitter import BaseSplitter
from chainer_chemistry.datasets import NumpyTupleDataset


class StratifiedSplitter(BaseSplitter):
    """Class for doing stratified data splits."""

    def _split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
               **kwargs):
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test,
                                          1.)

        seed = kwargs.get('seed')
        # TODO: feature?
        labels_feature_id = kwargs.get('labels_feature_id', -1)
        task_id = kwargs.get('task_id', 0)

        if not isinstance(dataset, NumpyTupleDataset):
            raise NotImplementedError
        labels_feature = dataset.features[:, labels_feature_id]
        if len(labels_feature.shape) == 1:
            labels = labels_feature
        else:
            labels = labels_feature[:, task_id]

        if labels.dtype.kind == 'i':
            classes, label_indices = numpy.unique(labels, return_inverse=True)
        elif labels.dtype.kind == 'f':
            n_bin = 10
            classes = numpy.arange(n_bin)
            label_indices = pandas.qcut(labels, n_bin, labels=False)
        else:
            raise ValueError

        n_classes = classes.shape[0]
        n_total_valid = int(numpy.floor(frac_valid * len(dataset)))
        n_total_test = int(numpy.floor(frac_test * len(dataset)))

        class_counts = numpy.bincount(label_indices)
        class_indices = numpy.split(numpy.argsort(label_indices,
                                                  kind='mergesort'),
                                    numpy.cumsum(class_counts)[:-1])

        n_valid_samples = _approximate_mode(class_counts, n_total_valid)
        class_counts = class_counts - n_valid_samples
        n_test_samples = _approximate_mode(class_counts, n_total_test)

        train_index = []
        valid_index = []
        test_index = []

        for i in range(n_classes):
            n_valid = n_valid_samples[i]
            n_test = n_test_samples[i]

            perm = numpy.random.RandomState(seed)\
                .permutation(len(class_indices[i]))
            class_perm_index = class_indices[i][perm]

            class_valid_index = class_perm_index[:n_valid]
            class_test_index = class_perm_index[n_valid:n_valid+n_test]
            class_train_index = class_perm_index[n_valid+n_test:]

            train_index.extend(class_train_index)
            valid_index.extend(class_valid_index)
            test_index.extend(class_test_index)

        assert n_total_valid == len(valid_index)
        assert n_total_test == len(test_index)

        return numpy.random.permutation(train_index),\
            numpy.random.permutation(valid_index),\
            numpy.random.permutation(test_index),


def _approximate_mode(class_counts, n_draws):
    n_class = len(class_counts)
    continuous = class_counts * n_draws / class_counts.sum()
    floored = numpy.floor(continuous)
    assert n_draws // n_class == floored.sum() // n_class
    # TODO: add remainder
    n_remainder = int(n_draws - floored.sum())
    remainder = continuous - floored
    inds = numpy.argsort(remainder)[:n_remainder]
    floored[inds] += 1
    assert n_draws == floored.sum()
    return floored.astype(numpy.int)
