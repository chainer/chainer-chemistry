import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.parsers import SDFFileParser
from chainer_chemistry.dataset.preprocessors.schnet_preprocessor import SchNetPreprocessor  # NOQA
from chainer_chemistry.datasets import get_tox21_filepath


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    return ret


@pytest.fixture
def pp():
    return SchNetPreprocessor()


def test_schnet_preprocessor(mol, pp):
    ret = pp.get_input_features(mol)
    assert len(ret) == 2
    actual_atom_array, actual_adj_array = ret

    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    numpy.testing.assert_array_equal(actual_atom_array, expect_atom_array)

    # TODO(nakago): write test for adj matrix.
    # print(actual_adj_array)
    # expect_adj_array = numpy.array([[1, 1, 0, 0],
    #                                 [1, 1, 1, 0],
    #                                 [0, 1, 1, 1],
    #                                 [0, 0, 1, 1]], dtype=numpy.float32)
    # numpy.testing.assert_array_equal(actual_adj_array, expect_adj_array)


@pytest.mark.slow
def test_schnet_preprocessor_with_tox21():
    preprocessor = SchNetPreprocessor()

    dataset = SDFFileParser(preprocessor, postprocess_label=None
                            ).parse(get_tox21_filepath('train'))

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (atom from, atom to)
    assert adjs.ndim == 2
    assert adjs.dtype == numpy.float32


def test_schnet_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        pp = SchNetPreprocessor(max_atoms=3, out_size=2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
