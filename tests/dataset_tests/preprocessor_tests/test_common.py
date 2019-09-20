import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.preprocessors import common
from chainer_chemistry.utils.extend import extend_adj
from chainer_chemistry.config import MAX_ATOMIC_NUM

@pytest.fixture
def sample_molecule():
    return Chem.MolFromSmiles('CN=C=O')


@pytest.fixture
def sample_molecule_2():
    return Chem.MolFromSmiles('Cc1ccccc1')


class TestGetAtomicNumbers(object):

    def test_normal(self, sample_molecule):
        actual = common.construct_atomic_number_array(sample_molecule)

        assert actual.shape == (4,)
        expect = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
        numpy.testing.assert_equal(actual, expect)

    def test_padding(self, sample_molecule):
        actual = common.construct_atomic_number_array(sample_molecule, 5)

        assert actual.shape == (5,)
        expect = numpy.array([6, 7, 6, 8, 0], dtype=numpy.int32)
        numpy.testing.assert_equal(actual, expect)

    def test_normal_truncated(self, sample_molecule):
        with pytest.raises(ValueError):
            adj = common.construct_atomic_number_array(sample_molecule, 3)  # NOQA


class TestGetAdjMatrix(object):

    def test_normal(self, sample_molecule_2):
        adj = common.construct_adj_matrix(sample_molecule_2)

        assert adj.shape == (7, 7)
        expect = numpy.array(
            [[1., 1., 0., 0., 0., 0., 0., ],
             [1., 1., 1., 0., 0., 0., 1., ],
             [0., 1., 1., 1., 0., 0., 0., ],
             [0., 0., 1., 1., 1., 0., 0., ],
             [0., 0., 0., 1., 1., 1., 0., ],
             [0., 0., 0., 0., 1., 1., 1., ],
             [0., 1., 0., 0., 0., 1., 1., ]],
            dtype=numpy.float32)
        numpy.testing.assert_equal(adj, expect)

    def test_normal_no_self_connection(self, sample_molecule_2):
        adj = common.construct_adj_matrix(sample_molecule_2,
                                          self_connection=False)

        assert adj.shape == (7, 7)
        expect = numpy.array(
            [[0., 1., 0., 0., 0., 0., 0.],
             [1., 0., 1., 0., 0., 0., 1.],
             [0., 1., 0., 1., 0., 0., 0.],
             [0., 0., 1., 0., 1., 0., 0.],
             [0., 0., 0., 1., 0., 1., 0.],
             [0., 0., 0., 0., 1., 0., 1.],
             [0., 1., 0., 0., 0., 1., 0.]],
            dtype=numpy.float32)
        numpy.testing.assert_equal(adj, expect)

    def test_normal_padding(self, sample_molecule_2):
        adj = common.construct_adj_matrix(sample_molecule_2, 8)

        assert adj.shape == (8, 8)
        expect = numpy.array(
            [[1., 1., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 0., 0., 0., 1., 0.],
             [0., 1., 1., 1., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0., 0., 0.],
             [0., 0., 0., 1., 1., 1., 0., 0.],
             [0., 0., 0., 0., 1., 1., 1., 0.],
             [0., 1., 0., 0., 0., 1., 1., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0.]],
            dtype=numpy.float32)
        numpy.testing.assert_equal(adj, expect)

    def test_normal_truncated(self, sample_molecule_2):
        with pytest.raises(ValueError):
            adj = common.construct_adj_matrix(sample_molecule_2, 6)


class TestConstructDiscreteEdgeMatrix(object):

    expect_adj = numpy.array(
            [[[0., 1., 0., 0., 0., 0., 0.],
              [1., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.]],
             [[0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.]],
             [[0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0.]],
             [[0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0., 1.],
              [0., 1., 0., 1., 0., 0., 0.],
              [0., 0., 1., 0., 1., 0., 0.],
              [0., 0., 0., 1., 0., 1., 0.],
              [0., 0., 0., 0., 1., 0., 1.],
              [0., 1., 0., 0., 0., 1., 0.]]], dtype=numpy.float32)

    def test_default(self, sample_molecule_2):
        adj = common.construct_discrete_edge_matrix(sample_molecule_2)
        assert adj.shape == (4, 7, 7)
        numpy.testing.assert_equal(adj, self.expect_adj)

    def test_add_self_connection_channel(self, sample_molecule_2):
        adj = common.construct_discrete_edge_matrix(
            sample_molecule_2, add_self_connection_channel=True)
        assert adj.shape == (5, 7, 7)
        numpy.testing.assert_equal(adj[:4], self.expect_adj)
        numpy.testing.assert_equal(adj[4], numpy.eye(7, 7))

    def test_padding(self, sample_molecule_2):
        adj = common.construct_discrete_edge_matrix(sample_molecule_2, 8)

        assert adj.shape == (4, 8, 8)
        expect = extend_adj(self.expect_adj, out_size=8, axis=[-1, -2])
        numpy.testing.assert_equal(adj, expect)

    def test_truncated(self, sample_molecule_2):
        with pytest.raises(ValueError):
            adj = common.construct_discrete_edge_matrix(sample_molecule_2, 6)  # NOQA


def test_construct_super_node_feature_adj_ndim2(sample_molecule):
    adj = common.construct_adj_matrix(sample_molecule)
    atom_array = common.construct_atomic_number_array(sample_molecule)
    s = common.construct_supernode_feature(sample_molecule, atom_array, adj)
    # print(s)
    assert s.shape == (MAX_ATOMIC_NUM * 2 + 4,)
    assert s[0] == len(atom_array)
    assert s[1] == adj.sum()
    assert s[2] == 1
    assert s[3] == 1
    assert s[3 + 6] == 1  # C
    assert s[3 + 7] == 1  # N
    assert s[3 + 8] == 1  # O
    assert s[3 + MAX_ATOMIC_NUM] == 0  # other
    assert s[3 + MAX_ATOMIC_NUM + 6] == 2 / len(atom_array)
    assert s[3 + MAX_ATOMIC_NUM + 7] == 1 / len(atom_array)
    assert s[3 + MAX_ATOMIC_NUM + 8] == 1 / len(atom_array)
    assert s[3 + MAX_ATOMIC_NUM * 2] == 0


def test_construct_super_node_feature_adj_ndim3(sample_molecule):
    adj = common.construct_discrete_edge_matrix(sample_molecule)
    atom_array = common.construct_atomic_number_array(sample_molecule)
    s = common.construct_supernode_feature(sample_molecule, atom_array, adj)
    assert s.shape == (MAX_ATOMIC_NUM * 2 + 10,)
    assert s[0] == len(atom_array)
    assert s[1] == adj.sum()
    assert s[2] == 1
    assert s[3] == 1
    assert s[4] == 0
    assert s[5] == 0
    assert pytest.approx(s[6], 1 * 2 / adj.sum()) # symmetric
    assert pytest.approx(s[7], 2 * 2 / adj.sum()) # symmetric
    assert s[8] == 0
    assert s[9] == 0
    assert s[9 + 6] == 1  # C
    assert s[9 + 6] == 1  # N
    assert s[9 + 7] == 1  # O
    assert s[9 + MAX_ATOMIC_NUM] == 0  # other
    assert s[9 + MAX_ATOMIC_NUM + 6] == 2 / len(atom_array)
    assert s[9 + MAX_ATOMIC_NUM + 7] == 1 / len(atom_array)
    assert s[9 + MAX_ATOMIC_NUM + 8] == 1 / len(atom_array)
    assert s[9 + MAX_ATOMIC_NUM * 2] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
