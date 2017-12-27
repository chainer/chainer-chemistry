import os

import numpy

from chainer.datasets import split_dataset_random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from chainer_chemistry.dataset.preprocessors import preprocess_method_dict
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

from chainer_chemistry.datasets.qm9 import get_qm9_label_names, get_qm9


def prepare_qm9_dataset(method, labels=None, train_data_ratio=0.7,
                        seed=777, scale='standardize', retain_smiles=False):
    if labels:
        cache_dir = os.path.join('input', '{}_{}'.format(method, labels))
        class_num = len(labels) if isinstance(labels, list) else 1
    else:
        labels = None
        cache_dir = os.path.join('input', '{}_all'.format(method))
        class_num = len(get_qm9_label_names())

    # Dataset preparation
    dataset = None

    #cache_dir = os.path.join('input', '{}'.format(method))
    if os.path.exists(cache_dir):
        print('load from cache {}'.format(cache_dir))
        dataset = NumpyTupleDataset.load(os.path.join(cache_dir, 'data.npz'))
    if dataset is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        if retain_smiles:
            dataset, smiles = get_qm9(preprocessor, labels=labels,
                                      retain_smiles=retain_smiles)
        else:
            dataset = get_qm9(preprocessor, labels=labels)

        os.makedirs(cache_dir)
        NumpyTupleDataset.save(os.path.join(cache_dir, 'data.npz'), dataset)

    test_size = 1 - train_data_ratio
    train_idx, val_idx = train_test_split(numpy.arange(len(dataset)),
                                          test_size=test_size,
                                          random_state=seed)
    ss = None
    if scale == 'standardize':
        # Standard Scaler for labels
        ss = StandardScaler()
        train_labels = dataset.features[train_idx, -1]
        val_labels = dataset.features[val_idx, -1]
        scaled_train_labels = ss.fit_transform(train_labels)
        scaled_val_labels = ss.transform(val_labels)
        # dataset = NumpyTupleDataset(*dataset.get_datasets()[:-1], labels)
        train = NumpyTupleDataset(*dataset.features[train_idx, :-1],
                                  scaled_train_labels)
        val = NumpyTupleDataset(*dataset.features[val_idx, :-1],
                                scaled_val_labels)
    else:
        train = NumpyTupleDataset(*dataset.features[train_idx, :])
        val = NumpyTupleDataset(*dataset.features[val_idx, :])

    if retain_smiles:
        train_smiles = smiles[train_idx]
        val_smiles = smiles[val_idx]
        return train, val, train_smiles, val_smiles, class_num, ss
    else:
        return train, val, class_num, ss
