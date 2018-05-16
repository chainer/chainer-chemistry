import os

import numpy

from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


class _CacheNamePolicy(object):

    train_file_name = 'train.npz'
    val_file_name = 'val.npz'
    test_file_name = 'test.npz'

    def _get_cache_directory_path(self, method, labels, prefix, num_data):
        num_data_str = '_{}'.format(num_data) if num_data >= 0 else ''
        if labels:
            return os.path.join(prefix,
                                '{}_{}{}'.format(method, labels, num_data_str))
        else:
            return os.path.join(prefix,
                                '{}_all{}'.format(method, num_data_str))

    def __init__(self, method, labels, prefix='input', num_data=-1):
        self.method = method
        self.labels = labels
        self.prefix = prefix
        self.num_data = num_data
        self.cache_dir = self._get_cache_directory_path(
            method, labels, prefix, num_data)

    def get_train_file_path(self):
        return os.path.join(self.cache_dir, self.train_file_name)

    def get_val_file_path(self):
        return os.path.join(self.cache_dir, self.val_file_name)

    def get_test_file_path(self):
        return os.path.join(self.cache_dir, self.test_file_name)

    def create_cache_directory(self):
        try:
            os.makedirs(self.cache_dir)
        except OSError:
            if not os.path.isdir(self.cache_dir):
                raise


def load_dataset(method, labels, prefix='input', num_data=-1):
    policy = _CacheNamePolicy(method, labels, prefix, num_data=num_data)
    train_path = policy.get_train_file_path()
    val_path = policy.get_val_file_path()
    test_path = policy.get_test_file_path()

    train, val, test = None, None, None
    print()
    if os.path.exists(policy.cache_dir):
        print('load from cache {}'.format(policy.cache_dir))
        train = NumpyTupleDataset.load(train_path)
        val = NumpyTupleDataset.load(val_path)
        test = NumpyTupleDataset.load(test_path)
    if train is None or val is None or test is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        if num_data >= 0:
            # Use `num_data` examples for train
            target_index = numpy.arange(num_data)
            train, val, test = D.get_tox21(
                preprocessor, labels=labels,
                train_target_index=target_index, val_target_index=None,
                test_target_index=None
            )
        else:
            train, val, test = D.get_tox21(preprocessor, labels=labels)
        # Cache dataset
        policy.create_cache_directory()
        NumpyTupleDataset.save(train_path, train)
        NumpyTupleDataset.save(val_path, val)
        NumpyTupleDataset.save(test_path, test)
    return train, val, test
