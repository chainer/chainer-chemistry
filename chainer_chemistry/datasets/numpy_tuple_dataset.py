import os
import six

import numpy

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.dataset.indexers.numpy_tuple_dataset_feature_indexer import NumpyTupleDatasetFeatureIndexer  # NOQA


class NumpyTupleDataset(object):

    """Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.
    And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.

    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
        self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, numpy.ndarray)):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length

    def get_datasets(self):
        return self._datasets

    @property
    def converter(self):
        return concat_mols

    @property
    def features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index

        .. admonition:: Example

           >>> import numpy
           >>> from chainer_chemistry.datasets import NumpyTupleDataset
           >>> x = numpy.array([0, 1, 2], dtype=numpy.float32)
           >>> t = x * x
           >>> numpy_tuple_dataset = NumpyTupleDataset(x, t)
           >>> targets = numpy_tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        return self._features_indexer

    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        numpy.savez(filepath, *numpy_tuple_dataset._datasets)

    @classmethod
    def load(cls, filepath, allow_pickle=True):
        if not os.path.exists(filepath):
            return None
        load_data = numpy.load(filepath, allow_pickle=allow_pickle)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return NumpyTupleDataset(*result)
