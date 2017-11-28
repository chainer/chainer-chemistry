import os
import numpy
import pytest
from rdkit import Chem
import six

from chainerchem.dataset.parsers import SDFFileParser
from chainerchem.dataset.preprocessors import NFPPreprocessor


@pytest.fixture
def mols():
    mol1 = Chem.MolFromSmiles('CN=C=O')
    mol2 = Chem.MolFromSmiles('Cc1ccccc1')
    return [mol1, mol2]


@pytest.fixture()
def sdf_file(tmpdir, mols):
    # Chem.AllChem.Compute2DCoords(mol1)
    fname = os.path.join(str(tmpdir), 'test.sdf')
    writer = Chem.SDWriter(fname)
    for mol in mols:
        writer.write(mol)
    return fname


def check_input_features(actual, expect):
    assert len(actual) == len(expect)
    for d, e in six.moves.zip(actual, expect):
        numpy.testing.assert_array_equal(d, e)


def test_sdf_file_parser(sdf_file, mols):
    preprocessor = NFPPreprocessor()
    parser = SDFFileParser(preprocessor)
    dataset = parser.parse(sdf_file)
    assert len(dataset) == 2

    # As we want test SDFFileParser, we assume
    # NFPPreprocessor works as documented.
    expect = preprocessor.get_input_features(mols[0])
    check_input_features(dataset[0], expect)

    expect = preprocessor.get_input_features(mols[1])
    check_input_features(dataset[1], expect)


# TODO(oono)
# test with non-default options of SDFFileParser


if __name__ == '__main__':
    pytest.main()
