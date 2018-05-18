import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.parsers.sdf_file_parser import SDFFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.datasets import get_tox21_filepath


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    return ret


def test_atomic_number_default_preprocessor(mol):
    preprocessor = AtomicNumberPreprocessor()
    ret_atom_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)


def test_atomic_number_non_default_padding_preprocessor(mol):
    preprocessor = AtomicNumberPreprocessor(out_size=10)
    ret_atom_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8, 0, 0, 0, 0, 0, 0],
                                    dtype=numpy.int32)
    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)


def test_atomic_number_non_default_max_atoms_preprocessor(mol):
    preprocessor = AtomicNumberPreprocessor(max_atoms=5)
    ret_atom_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8],
                                    dtype=numpy.int32)
    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)

    preprocessor = AtomicNumberPreprocessor(max_atoms=3)
    with pytest.raises(MolFeatureExtractionError):
        preprocessor.get_input_features(mol)


def test_atomic_number_preprocessor(mol):
    preprocessor = AtomicNumberPreprocessor(max_atoms=5, out_size=10)
    ret_atom_array = preprocessor.get_input_features(mol)
    expect_atom_array = numpy.array([6, 7, 6, 8, 0, 0, 0, 0, 0, 0],
                                    dtype=numpy.int32)
    numpy.testing.assert_array_equal(ret_atom_array, expect_atom_array)


@pytest.mark.slow
def test_atomic_number_preprocessor_with_tox21():
    preprocessor = AtomicNumberPreprocessor()
    dataset = SDFFileParser(preprocessor) \
        .parse(get_tox21_filepath('train'))['dataset']
    index = numpy.random.choice(len(dataset), None)
    atoms, = dataset[index]

    assert atoms.ndim == 1
    assert atoms.dtype == numpy.int32


def test_atomic_number_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        AtomicNumberPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main()
