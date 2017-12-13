import os

from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry import datasets as D
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


class _CacheNamePolicy(object):

    train_file_name = 'train.npz'
    val_file_name = 'val.npz'
    test_file_name = 'test.npz'

    def _get_cache_directory_path(self, method, labels, prefix):
        if labels:
            return os.path.join(prefix, '{}_{}'.format(method, labels))
        else:
            return os.path.join(prefix, '{}_all'.format(method))

    def __init__(self, method, labels, prefix='input'):
        self.method = method
        self.labels = labels
        self.prefix = prefix
        self.cache_dir = self._get_cache_directory_path(method, labels, prefix)

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


def load_dataset(method, labels, prefix='input'):
    policy = _CacheNamePolicy(method, labels, prefix)
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
        train, val, test = D.get_tox21(preprocessor, labels=labels)
        # Cache dataset
        policy.create_cache_directory()
        NumpyTupleDataset.save(train_path, train)
        NumpyTupleDataset.save(val_path, val)
        NumpyTupleDataset.save(test_path, test)
    return train, val, test
