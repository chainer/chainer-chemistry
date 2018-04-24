import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.dataset.parsers.sdf_file_parser import SDFFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import get_tox21_filepath


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    return ret


@pytest.fixture
def pp():
    return AtomicNumberPreprocessor()


def test_atomic_number_preprocessor(mol, pp):
    ret = pp.get_input_features(mol)
    print(ret, type(ret), ret.shape, ret.dtype)
    # assert len(ret) == 1
    actual_atom_array = ret

    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    numpy.testing.assert_array_equal(actual_atom_array, expect_atom_array)


# TODO(Oono)
# Test non-default max_atom and non-default zero_padding options, respectively
# after the discussion of the issue #60.


@pytest.mark.slow
def test_atomic_number_preprocessor_with_tox21():
    preprocessor = AtomicNumberPreprocessor()

    # labels=None as default, and label information is not returned.
    dataset = SDFFileParser(preprocessor)\
        .parse(get_tox21_filepath('train'))['dataset']
    index = numpy.random.choice(len(dataset), None)
    atoms, = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32


def test_atomic_number_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        pp = AtomicNumberPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main()
