import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.parsers.sdf_file_parser import SDFFileParser
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.rsgcn_preprocessor import RSGCNPreprocessor  # NOQA
from chainer_chemistry.datasets import get_tox21_filepath


@pytest.fixture
def mol():
    return Chem.MolFromSmiles('CN=C=O')


def test_rsgcn_default_preprocessor(mol):
    preprocessor = RSGCNPreprocessor()
    ret_atom_array, ret_adj_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    expect_adj_array = numpy.array(
        [[0.5, 0.4082, 0, 0], [0.4082, 0.3333, 0.3333, 0],
         [0, 0.3333, 0.3333, 0.4082], [0, 0, 0.4082, 0.5]],
        dtype=numpy.float32)

    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)
    numpy.testing.assert_allclose(
        ret_adj_array, expect_adj_array, rtol=1e-03, atol=1e-03)


def test_rsgcn_non_default_padding_preprocessor(mol):
    preprocessor = RSGCNPreprocessor(out_size=7)
    ret_atom_array, ret_adj_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8, 0, 0, 0], dtype=numpy.int32)
    expect_adj_array = numpy.array(
        [[0.5, 0.4082, 0, 0, 0, 0, 0], [0.4082, 0.3333, 0.3333, 0, 0, 0, 0],
         [0, 0.3333, 0.3333, 0.4082, 0, 0, 0], [0, 0, 0.4082, 0.5, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        dtype=numpy.float32)

    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)
    numpy.testing.assert_allclose(
        ret_adj_array, expect_adj_array, rtol=1e-03, atol=1e-03)


def test_rsgcn_non_default_max_atoms_preprocessor(mol):
    preprocessor = RSGCNPreprocessor(max_atoms=5)
    ret_atom_array, ret_adj_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    expect_adj_array = numpy.array(
        [[0.5, 0.4082, 0, 0], [0.4082, 0.3333, 0.3333, 0],
         [0, 0.3333, 0.3333, 0.4082], [0, 0, 0.4082, 0.5]],
        dtype=numpy.float32)

    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)
    numpy.testing.assert_allclose(
        ret_adj_array, expect_adj_array, rtol=1e-03, atol=1e-03)

    preprocessor = RSGCNPreprocessor(max_atoms=3)
    with pytest.raises(MolFeatureExtractionError):
        preprocessor.get_input_features(mol)


def test_rsgcn_preprocessor(mol):
    preprocessor = RSGCNPreprocessor(max_atoms=4, out_size=4)
    ret_atom_array, ret_adj_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    expect_adj_array = numpy.array(
        [[0.5, 0.4082, 0, 0], [0.4082, 0.3333, 0.3333, 0],
         [0, 0.3333, 0.3333, 0.4082], [0, 0, 0.4082, 0.5]],
        dtype=numpy.float32)

    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)
    numpy.testing.assert_allclose(
        ret_adj_array, expect_adj_array, rtol=1e-03, atol=1e-03)


@pytest.mark.slow
def test_rsgcn_preprocessor_with_tox21():
    preprocessor = RSGCNPreprocessor()

    # labels=None as default, and label information is not returned.
    dataset = SDFFileParser(preprocessor)\
        .parse(get_tox21_filepath('train'))['dataset']
    index = numpy.random.choice(len(dataset), None)
    atoms, adjacency = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    assert adjacency.ndim == 2
    assert adjacency.dtype == numpy.float32


def test_rsgcn_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        RSGCNPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main()
