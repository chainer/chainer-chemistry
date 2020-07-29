import numpy as np
import pytest

from chainer_chemistry.dataset.preprocessors import wle_util


def test_to_index():
    values = ['foo', 'bar', 'buz', 'non-exist']
    mols = [['foo', 'bar', 'buz'], ['foo', 'foo'], ['buz', 'bar']]

    actual = wle_util.to_index(mols, values)
    expect = np.array([np.array([0, 1, 2], np.int32),
                       np.array([0, 0], np.int32),
                       np.array([2, 1], np.int32)])
    assert len(actual) == len(expect)
    for a, e in zip(actual, expect):
        np.testing.assert_array_equal(a, e)


def test_to_index_non_existence():
    values = ['foo', 'bar']
    mols = [['strange_label']]

    with pytest.raises(ValueError):
        wle_util.to_index(mols, values)


def test_compress_relation_axis_2_dim():
    arr = np.random.uniform(size=(10, 2))
    actual = wle_util.compress_relation_axis(arr)
    np.testing.assert_array_equal(actual, arr)


def test_compress_relation_axis_3_dim():
    arr = np.array(
        [
            [
                [1, 0],
                [2, 0],
            ],
            [
                [1, 1],
                [0, 0]
            ]
        ]
    )
    arr = np.swapaxes(arr, 0, 1)
    ret = wle_util.compress_relation_axis(arr)
    actual = ret != 0
    expect = np.array(
        [[True, True],
        [True, False]]
    )
    np.testing.assert_array_equal(actual, expect)


def test_compress_relation_axis_invalid_ndim():
    arr = np.zeros(3)
    with pytest.raises(ValueError):
        wle_util.compress_relation_axis(arr)

    arr = np.zeros((1, 2, 3, 4))
    with pytest.raises(ValueError):
        wle_util.compress_relation_axis(arr)


@pytest.fixture
def small_molecule():
    # a-b-c d
    atom_array = ['a', 'b', 'c', 'd']
    neighbors = np.array(
        [
            [0, 1, 1, 2],  # first end of edges
            [1, 0, 2, 1]   # second end of edges
        ]
    )
    return atom_array, neighbors


def test_get_neighbor_representation_with_focus_atom(small_molecule):
    atom_array, neighbors = small_molecule
    expects = ['a-b', 'b-a.c', 'c-b', 'd-']
    for i in range(len(expects)):
        actual = wle_util.get_neighbor_representation(
            i, atom_array, neighbors, True)
        assert actual == expects[i]


def test_get_neighbor_representation_without_focus_atom(small_molecule):
    atom_array, neighbors = small_molecule
    expects = ['b', 'a.c', 'b', '']
    for i in range(len(expects)):
        actual = wle_util.get_neighbor_representation(
            i, atom_array, neighbors, False)
        assert actual == expects[i]


@pytest.mark.parametrize('label, expect', [
    ('a-b', 'a'),
    ('a-b.c', 'a'),
    ('aa-b', 'aa'),
    ('a-', 'a'),
    ('aa-', 'aa'),
])
def test_get_focus_node_label(label, expect):
    actual = wle_util.get_focus_node_label(label)
    assert actual == expect


@pytest.mark.parametrize('label', ['aa', 'a-a-a', 'a--'])
def test_get_focus_node_label_invalid(label):
    with pytest.raises(ValueError):
        wle_util.get_focus_node_label(label)
