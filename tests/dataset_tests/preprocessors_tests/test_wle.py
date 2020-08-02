import numpy as np
import pytest

from chainer_chemistry.dataset.preprocessors import wle as WLE  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def small_datasets():
    N_1 = 3
    N_2 = 5

    # one-hot atom labels: 1 tp N
    atom_array_1 = np.arange(N_1)
    atom_array_2 = np.arange(N_2)

    # adj-array, manually
    # all connectes. expanded labels is a permutaion of 0,1,2
    adj_array_1 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]).astype(np.int32)
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


def _get_elements(datasets, idx):
    return [[mol[1] for mol in d] for d in datasets]


def _get_atom_arrays(datasets):
    return _get_elements(datasets, 0)


def _get_adj_arrays(datasets):
    return _get_elements(datasets, 1)


def _get_wle_arrays(datasets):
    return _get_elements(datasets, 2)


def _get_teach_signals(datasets, is_cwle=False):
    if is_cwle:
        return _get_elements(datasets, 2)
    else:
        return _get_elements(datasets, 3)


def _check_np_array(actuals, expects):
    assert len(actuals) == len(expects) == 3  # train/test/val
    for actual_adjs, expect_adjs in zip(actuals, expects):
        assert len(actual_adjs) == len(expect_adjs)
        [np.testing.assert_array_equal(a, e)
            for a, e in zip(actual_adjs, expect_adjs)]


def test_wle(small_datasets):
    ret_value = WLE.apply_wle_for_datasets(small_datasets, 0)
    actual_datasets, actual_labels, actual_frequency = ret_value

    expected_frequency = {'0-1.2': 3,
                          '1-0.2': 3,
                          '2-0.1': 3,
                          '0-1.4': 3,
                          '1-0.4': 3,
                          '2-3': 3,
                          '3-2': 3,
                          '4-0.1': 3}
    assert expected_frequency == actual_frequency

    expected_labels = set(expected_frequency.keys())
    assert expected_labels == set(actual_labels)

    actual_adj_arrays = _get_adj_arrays(actual_datasets)
    expect_adj_arrays = _get_adj_arrays(small_datasets)
    _check_np_array(actual_adj_arrays, expect_adj_arrays)

    actual_signal_arrays = _get_teach_signals(actual_datasets)
    expect_signal_arrays = _get_teach_signals(small_datasets)
    _check_np_array(actual_signal_arrays, expect_signal_arrays)

    # Check atom_arrays of train/val/test datasets are identical.
    # 2 is the number of samples in each (train/val/test) dataset.
    atom_arrays = _get_atom_arrays(actual_datasets)
    first_mols = [d[0] for d in atom_arrays]
    second_mols = [d[1] for d in atom_arrays]
    for mols in (first_mols, second_mols):
        assert len(mols) == 3
        np.testing.assert_array_equal(mols[0], mols[1])
        np.testing.assert_array_equal(mols[1], mols[2])


def test_2_hop_wle(small_datasets):
    k = 2
    ret_value = WLE.apply_wle_for_datasets(small_datasets, 0, k)
    actual_datasets, actual_labels, actual_frequency = ret_value

    expected_frequency = {'0-1.2': 3,
                          '1-0.2': 3,
                          '2-0.1': 3,
                          '3-4.7': 3,
                          '4-3.7': 3,
                          '5-6': 3,
                          '6-5': 3,
                          '7-3.4': 3}
    # Kenta Oono (oono@preferred.jp)
    # The following assertion checks too strong condition.
    # Specifically it assumes that the WLE algorithm assigns
    # the extended atom labels appeared in the first iteration
    # in a certain order and runs the second iteration.
    # Strictly speaking, this is not required in the algorithm.
    assert expected_frequency == actual_frequency

    expected_labels = set(expected_frequency.keys())
    assert expected_labels == set(actual_labels)

    actual_adj_arrays = _get_adj_arrays(actual_datasets)
    expect_adj_arrays = _get_adj_arrays(small_datasets)
    _check_np_array(actual_adj_arrays, expect_adj_arrays)

    actual_signal_arrays = _get_teach_signals(actual_datasets)
    expect_signal_arrays = _get_teach_signals(small_datasets)
    _check_np_array(actual_signal_arrays, expect_signal_arrays)

    # Check atom_arrays of train/val/test datasets are identical.
    # 2 is the number of samples in each (train/val/test) dataset.
    atom_arrays = _get_atom_arrays(actual_datasets)
    first_mols = [d[0] for d in atom_arrays]
    second_mols = [d[1] for d in atom_arrays]
    for mols in (first_mols, second_mols):
        assert len(mols) == 3
        np.testing.assert_array_equal(mols[0], mols[1])
        np.testing.assert_array_equal(mols[1], mols[2])


def test_cwle(small_datasets):
    ret_value = WLE.apply_cwle_for_datasets(small_datasets)
    actual_datasets, actual_labels, actual_frequency = ret_value

    expected_frequency = {'1.2': 3,
                          '0.2': 3,
                          '0.1': 6,
                          '1.4': 3,
                          '0.4': 3,
                          '3': 3,
                          '2': 3}
    assert expected_frequency == actual_frequency

    expected_labels = set(expected_frequency.keys())
    assert expected_labels == set(actual_labels)

    actual_adj_arrays = _get_adj_arrays(actual_datasets)
    expect_adj_arrays = _get_adj_arrays(small_datasets)
    _check_np_array(actual_adj_arrays, expect_adj_arrays)

    actual_signal_arrays = _get_teach_signals(actual_datasets, True)
    expect_signal_arrays = _get_teach_signals(small_datasets)
    _check_np_array(actual_signal_arrays, expect_signal_arrays)

    # Check atom_arrays of train/val/test datasets are identical.
    atom_arrays = _get_atom_arrays(actual_datasets)
    first_mols = [d[0] for d in atom_arrays]
    second_mols = [d[1] for d in atom_arrays]
    for mols in (first_mols, second_mols):
        assert len(mols) == 3
        np.testing.assert_array_equal(mols[0], mols[1])
        np.testing.assert_array_equal(mols[1], mols[2])

    # Check wle_arrays of train/val/test datasets are identical.
    wle_arrays = _get_wle_arrays(actual_datasets)
    first_mols = [d[0] for d in wle_arrays]
    second_mols = [d[1] for d in wle_arrays]
    for mols in [first_mols, second_mols]:
        assert len(mols) == 3
        np.testing.assert_array_equal(mols[0], mols[1])
        np.testing.assert_array_equal(mols[1], mols[2])


def test_findmaxidx_atom_label(small_datasets):
    actual = WLE.findmaxidx(small_datasets, 'atom_label')
    expect = 5
    assert actual == expect


@pytest.fixture
def cwle_datasets():
    B = 10
    D_atom = 5
    D_wle = 50
    K_large = 10000

    atom_arrays = [np.full((B, D_atom), K_large) for _ in range(3)]
    adj_arrays = [np.eye(B, dtype=np.int32) for _ in range(3)]
    wle_arrays = [np.arange(B * D_wle, dtype=np.int32).reshape(B, -1)
                  for _ in range(3)]
    signal_arrays = [np.full(B, K_large) for _ in range(3)]

    print(wle_arrays[0].shape)

    datasets = [NumpyTupleDataset(atom_arrays[i],
                                  adj_arrays[i],
                                  wle_arrays[i],
                                  signal_arrays[i])
                for i in range(3)]
    return datasets


def test_findmaxidx_wle(cwle_datasets):
    actual = WLE.findmaxidx(cwle_datasets, 'wle_label')
    expect = 10 * 50
    assert actual == expect
