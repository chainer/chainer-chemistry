import itertools

import numpy as np
import pytest

from chainer_chemistry.dataset.preprocessors import wle_atom_array_update as wle_update
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def k3_datasets():
    train_atoms = np.array([np.zeros(3, dtype=np.int32)])
    val_atoms = np.array([np.ones(3, dtype=np.int32)])
    test_atoms = np.array([np.full(3, 2, dtype=np.int32)])

    train_adjs = np.array([np.ones((3, 3), dtype=np.int32)])
    val_adjs = np.array([np.ones((3, 3), dtype=np.int32)])
    test_adjs = np.array([np.ones((3, 3), dtype=np.int32)])
    return ((train_atoms, val_atoms, test_atoms),
            (train_adjs, val_adjs, test_adjs))


def _is_all_same(arr):
    arr = np.array(arr)
    assert arr.size > 0
    return np.all(arr == arr.item(0))


def _is_all_different(arr):
    for x, y in itertools.combinations(arr, 2):
        if x == y:
            return False
    return True


@pytest.mark.parametrize('cutoff', (0, 1, 2, 3, 4))
def test_update_atom_array(k3_datasets, cutoff):
    atom_arrays, adj_arrays = k3_datasets
    actual_atom_arrays, actual_label_frequency = wle_update.update_atom_arrays(
        atom_arrays, adj_arrays, cutoff)

    mols = [d[0] for d in actual_atom_arrays]
    for m in mols:
        assert _is_all_same(m)

    # train/val/test atoms must have different labels.
    assert _is_all_different((mols[0][0], mols[1][0], mols[2][0]))

    if cutoff >= 3:
        expect_label_frequency = {'0': 3, '1': 3, '2': 3}
    else:
        expect_label_frequency = {'0-0.0': 3, '1-1.1': 3, '2-2.2': 3}
    assert actual_label_frequency == expect_label_frequency


@pytest.fixture
def single_atom_datasets():
    train_atoms = np.array([[0], [1], [2]], dtype=np.int32)
    val_atoms = np.array([[1], [1], [5]], dtype=np.int32)
    test_atoms = np.array([[4], [4], [2]], dtype=np.int32)

    train_adjs = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)
    val_adjs = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)
    test_adjs = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)   
    return ((train_atoms, val_atoms, test_atoms),
            (train_adjs, val_adjs, test_adjs))


@pytest.mark.parametrize('cutoff', (0, 1, 2))
def test_update_atom_array_2(single_atom_datasets, cutoff):
    atom_arrays, adj_arrays = single_atom_datasets
    actual_atom_arrays, actual_label_frequency = wle_update.update_atom_arrays(
        atom_arrays, adj_arrays, cutoff)

    # Note that labels after expansion need not
    # same as the original atom labels.
    # For example, assigning ids accoring to
    # appearance order
    # 0 -> 0, 1 -> 1, 2 -> 2, 5 -> 3, 4 -> 4,
    # results in 
    # Atom arrays
    #   train: [[0], [1], [2]]
    #   val:   [[1], [1], [3]]
    #   test:  [[4], [4], [2]]
    # Label Frequency
    #    {'0': 1, '1': 3, '2': 2, '3': 1, '4': 2}
    # This is acceptable.

    train, val, test =  actual_atom_arrays
    assert _is_all_same((train[1], val[0], val[1]))
    assert _is_all_same((train[2], test[2]))
    assert _is_all_same((test[0], test[1]))
    assert _is_all_different((train[0], train[1], train[2], val[2], test[0]))

    expect_label_frequency = {'0-': 1, '1-': 3, '2-': 2, '4-': 2, '5-': 1}
    # Equal as a multiset.
    assert (sorted(actual_label_frequency.values())
             == sorted(expect_label_frequency.values()))


@pytest.fixture
def different_sample_size_datasets():
    train_atoms = np.array([[0]], dtype=np.int32)
    val_atoms = np.array([[0], [0]], dtype=np.int32)
    test_atoms = np.array([[0], [0], [0]], dtype=np.int32)

    train_adjs = np.array([[[1]]], dtype=np.int32)
    val_adjs = np.array([[[1]], [[1]]], dtype=np.int32)
    test_adjs = np.array([[[1]], [[1]], [[1]]], dtype=np.int32)
    return ((train_atoms, val_atoms, test_atoms),
            (train_adjs, val_adjs, test_adjs))


def test_update_atom_array_with_diffent_sample_sizes(
    different_sample_size_datasets):
    atom_arrays, adj_arrays = different_sample_size_datasets
    actual_atom_arrays, actual_label_frequency = wle_update.update_atom_arrays(
        atom_arrays, adj_arrays, 0)

    all_atoms = sum([list(a.ravel()) for a in actual_atom_arrays], [])
    assert _is_all_same(all_atoms)

    expect_label_frequency = {'0-': 6}
    assert actual_label_frequency == expect_label_frequency


@pytest.fixture
def different_graph_size_datasets():
    train_atoms = np.array([[0]], dtype=np.int32)
    val_atoms = np.array([[0, 0]], dtype=np.int32)
    test_atoms = np.array([[0, 0, 0]], dtype=np.int32)

    train_adjs = np.array([[[1]]], dtype=np.int32)
    val_adjs = np.array([[[1, 1],
                          [1, 1]]], dtype=np.int32)
    test_adjs = np.array([[[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]], dtype=np.int32)
    return ((train_atoms, val_atoms, test_atoms),
            (train_adjs, val_adjs, test_adjs))


def test_update_atom_array_with_different_graph_size(
    different_graph_size_datasets):
    atom_arrays, adj_arrays = different_graph_size_datasets
    actual_atom_arrays, actual_label_frequency = wle_update.update_atom_arrays(
        atom_arrays, adj_arrays, 0)

    mols = [d[0] for d in actual_atom_arrays]
    for m in mols:
        assert _is_all_same(m)

    expect_label_frequency = {'0-': 1, '0-0': 2, '0-0.0': 3}
    assert actual_label_frequency == expect_label_frequency


@pytest.fixture
def line_graph_datasets():
    train_atoms = np.zeros(5, dtype=np.int32).reshape(1, -1)
    val_atoms = np.array([[1]], dtype=np.int32)
    test_atoms = np.array([[1]], dtype=np.int32)

    train_adjs = np.array([[[1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1],
                            [0, 0, 0, 1, 1]]],
                            dtype=np.int32)
    val_adjs = np.array([[[1]]], dtype=np.int32)
    test_adjs = np.array([[[1]]], dtype=np.int32)
    return ((train_atoms, val_atoms, test_atoms),
            (train_adjs, val_adjs, test_adjs))


def test_update_atom_array_twice(line_graph_datasets):
    atom_arrays, adj_arrays = line_graph_datasets

    for _ in range(2):
        atom_arrays, actual_label_frequency = wle_update.update_atom_arrays(
            atom_arrays, adj_arrays, 0)

    expect_label_frequency = {'0-1': 2,
                              '1-0.1': 2,
                              '1-1.1': 1,
                              '2-': 2}    # atoms in test and val datasets
    assert actual_label_frequency == expect_label_frequency


@pytest.fixture
def small_datasets():
    N_1 = 3
    N_2 = 5

    # one-hot atom labels: 1 tp N
    atom_array_1 = np.arange(N_1)
    atom_array_2 = np.arange(N_2)

    # adj-array, manually
    # all connectes. expanded labels is a permutaion of 0,1,2
    adj_array_1 = np.ones((3, 3), dtype=np.int32)
    # node 0 --> 0-1.2
    # node 1 --> 1-0.2
    # node 2 --> 2-0.1

    adj_array_2 = np.array([[1, 1, 0, 0, 1],
                            [1, 1, 0, 0, 1],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [1, 1, 0, 0, 1]]).astype(np.float32)
    # node 0 --> 0-1.4
    # node 1 --> 1-0.4
    # node 2 --> 2-3
    # node 3 --> 3-2
    # node 4 --> 4-0.1

    # supervised labels, dummy
    teach_signal_1 = np.array(1).astype(np.int)
    teach_signal_2 = np.array(0).astype(np.int)

    # concat in a one numpy array!
    atom_arrays = np.array([atom_array_1, atom_array_2])
    adj_arrays = np.array([adj_array_1, adj_array_2])
    teach_signals = np.array([teach_signal_1, teach_signal_2])

    # train/val/test dataset, respectively
    datasets = [NumpyTupleDataset(atom_arrays, adj_arrays, teach_signals),
                NumpyTupleDataset(atom_arrays, adj_arrays, teach_signals),
                NumpyTupleDataset(atom_arrays, adj_arrays, teach_signals)]
    return datasets


def test_list_all_expanded_labels_with_focus_atom(small_datasets):
    atom_arrays = [[mol[0] for mol in d] for d in small_datasets]
    adj_arrays = [[mol[1] for mol in d] for d in small_datasets]

    actual_atom_lists, actual_frequencies = wle_update.list_all_expanded_labels(
        atom_arrays, adj_arrays, True)

    expected_frequency = {'0-1.2': 3,
                          '1-0.2': 3,
                          '2-0.1': 3,
                          '0-1.4': 3,
                          '1-0.4': 3,
                          '2-3': 3,
                          '3-2': 3,
                          '4-0.1': 3}
    assert expected_frequency == actual_frequencies

    expect_atom_list = [
        set(['0-1.2', '1-0.2', '2-0.1']),
        set(['0-1.4', '1-0.4', '2-3', '3-2', '4-0.1'])]
    for actual_atom_list in actual_atom_lists:
        for a, e in zip(actual_atom_list, expect_atom_list):
            assert set(a) == e


def test_list_all_expanded_labels_without_focus_atom(small_datasets):
    atom_arrays = [[mol[0] for mol in d] for d in small_datasets]
    adj_arrays = [[mol[1] for mol in d] for d in small_datasets]
    actual_atom_lists, actual_frequencies = wle_update.list_all_expanded_labels(
        atom_arrays, adj_arrays, False)

    expected_frequency = {'1.2': 3,
                          '0.2': 3,
                          '0.1': 6,
                          '1.4': 3,
                          '0.4': 3,
                          '3': 3,
                          '2': 3}
    assert expected_frequency == actual_frequencies

    expect_atom_list = [
        set(['1.2', '0.2', '0.1']),
        set(['1.4', '0.4', '3', '2', '0.1'])]
    for actual_atom_list in actual_atom_lists:
        for a, e in zip(actual_atom_list, expect_atom_list):
            assert set(a) == e
