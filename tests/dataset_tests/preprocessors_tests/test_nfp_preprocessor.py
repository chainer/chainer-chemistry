import os
import numpy
import pytest
from rdkit import Chem

from chainerchem.dataset.parsers import SDFFileParser
from chainerchem.dataset.preprocessors import NFPPreprocessor
from chainerchem.datasets import get_tox21_filepath


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    return ret


@pytest.fixture
def pp():
    return NFPPreprocessor()


def test_nfp_preprocessor(mol, pp):
    ret = pp.get_input_features(mol)
    assert len(ret) == 2
    actual_atom_array, actual_adj_array = ret

    expect_atom_array = numpy.array([6, 7, 6, 8], dtype=numpy.int32)
    numpy.testing.assert_array_equal(actual_atom_array, expect_atom_array)

    expect_adj_array = numpy.array([[1, 1, 0, 0],
                                    [1, 1, 1, 0],
                                    [0, 1, 1, 1],
                                    [0, 0, 1, 1]], dtype=numpy.float32)
    numpy.testing.assert_array_equal(actual_adj_array, expect_adj_array)


# TODO (Oono)
# Test non-default max_atom and non-default zero_padding options, respectively
# after the discussion of the issue #60.


@pytest.mark.slow
def test_nfp_preprocessor_with_tox21():
    preprocessor = NFPPreprocessor()

    dataset = SDFFileParser(preprocessor, postprocess_label=None
                            ).parse(get_tox21_filepath('train'))

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (atom from, atom to) or (edge_type, atom from, atom to)
    # TODO: Spec not fixed yet...
    assert adjs.ndim == 2 or adjs.ndim == 3
    assert adjs.dtype == numpy.float32


if __name__ == '__main__':
    pytest.main()
