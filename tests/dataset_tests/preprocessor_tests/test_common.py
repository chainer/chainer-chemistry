import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.preprocessors import common


@pytest.fixture
def sample_molecule():
    return Chem.MolFromSmiles('CN=C=O')


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


@pytest.fixture
def sample_molecule_2():
    return Chem.MolFromSmiles('Cc1ccccc1')


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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
