import collections

import numpy as np

from chainer_chemistry.dataset.preprocessors import wle_io
from chainer_chemistry.dataset.preprocessors import wle_atom_array_update as wle_update

DEBUG = False

def apply_wle_for_datasets(datasets, cutoff=0, k=1):
    """
    Apply label Weisfeiler--Lehman Embedding for the tuple of datasets.

    Args:
        datasets: tuple of dataset (usually, train/val/test),
                     each dataset consists of atom_array and
                     adj_array and teach_signal
        cutoff: int, if more than 0, the expanded labels
                   whose freq <= cutoff will be removed.
        k: int, the number of iterations of neighborhood
              aggregation.

    Returns:
        - tuple of dataset (usually, train/val/test),
               each dataest consists of atom_number_array and
               adj_tensor with expanded labels
        - list of all labels, used in the dataset parts.
        - dictionary of label frequencies key:label valeu:frequency count
    """

    atom_arrays, adj_arrays, teach_signals = wle_io.load_dataset_elements(datasets)

    for _ in range(k):
        atom_arrays, labels_frequencies = wle_update.update_atom_arrays(
            atom_arrays, adj_arrays, cutoff)

    datasets_expanded = wle_io.create_datasets(atom_arrays, adj_arrays, teach_signals)
    expanded_labels = list(labels_frequencies.keys())
    return tuple(datasets_expanded), expanded_labels, labels_frequencies


def apply_cwle_for_datasets(datasets, k=1):
    """
    Apply Concatenated Weisfeiler--Lehman embedding for the tuple of datasets.
    This also applicalbe for the Gated-sum Weisfeiler--Lehman embedding.

    Args:
        datasets: tuple of dataset (usually, train/val/test),
                     each dataset consists of atom_array and
                     adj_array and teach_signal
        k: int, the number of iterations of neighborhood
              aggregation.

    Returns:
        - tuple of dataset (usually, train/val/test),
               each dataest consists of atom_number_array,
               expanded_label_array, and adj_tensor
        - list of all expanded labels, used in the dataset parts.
        - dictionary of label frequencies key:label valeu:frequency count
    """

    if k <= 0:
        raise ValueError('Iterations should be a positive integer. '
                         'Found k={}'.format(k))

    atom_arrays, adj_arrays, teach_signals = wle_io.load_dataset_elements(datasets)

    for i in range(k):
        if i != k - 1:
            atom_arrays, labels_frequencies = wle_update.update_atom_arrays(
                atom_arrays, adj_arrays, 0)
        else:
            wle_arrays, labels_frequencies = wle_update.update_atom_arrays(
                atom_arrays, adj_arrays, 0, False)

    datasets_expanded = wle_io.create_datasets(
        atom_arrays, adj_arrays, teach_signals, wle_arrays)
    expanded_labels = list(labels_frequencies.keys())
    return tuple(datasets_expanded), expanded_labels, labels_frequencies

def _findmaxidx(datasets, idx):
    atom_data_size = len(datasets[0][0])
    if atom_data_size <= idx:
        raise ValueError(
            'data index is out of index. '
            'atom_data size={} <= idx={}'.format(
                atom_data_size, idx))

    max_idx = -1
    for dataset in datasets:
        for mol_data in dataset:
            atom_array = mol_data[idx]
            max_atom_idx = np.max(atom_array)
            if max_atom_idx > max_idx:
                max_idx = max_atom_idx

    return max_idx + 1  # 0-origin

def findmaxidx(datasets, target='atom_label'):
    """
    Retruns the maximum number of the symbol index in an atom array,
    throughout the datasets.

    Args:
        datasets: dataset entity
        target: choice of 'atom_label' of 'wle_label'

    Returns:
        _findmaxidx(datasets, 0/2)
    """

    if target == 'atom_label':
        return _findmaxidx(datasets, 0)
    elif target == 'wle_label':
        return _findmaxidx(datasets, 2)

