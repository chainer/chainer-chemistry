import numpy
import pytest
from rdkit import Chem
import six

from chainer_chemistry.dataset.parsers import SmilesParser
from chainer_chemistry.dataset.preprocessors import NFPPreprocessor


@pytest.fixture
def mol_smiles():
    mol_smiles1 = 'CN=C=O'
    mol_smiles2 = 'Cc1ccccc1'
    mol_smiles3 = 'CC1=CC2CC(CC1)O2'
    return [mol_smiles1, mol_smiles2, mol_smiles3]


@pytest.fixture
def mols(mol_smiles):
    return [Chem.MolFromSmiles(smiles) for smiles in mol_smiles]


def check_input_features(actual, expect):
    assert len(actual) == len(expect)
    for d, e in six.moves.zip(actual, expect):
        numpy.testing.assert_array_equal(d, e)


def test_smiles_parser_not_return_smiles(mol_smiles, mols):
    preprocessor = NFPPreprocessor()
    parser = SmilesParser(preprocessor)
    result = parser.parse(mol_smiles, return_smiles=False)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3
    assert smiles is None

    # As we want test CSVFileParser, we assume
    # NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_input_features(dataset[i], expect)


def test_smiles_parser_return_smiles(mol_smiles, mols):
    """test `labels` option and retain_smiles=True."""
    preprocessor = NFPPreprocessor()
    parser = SmilesParser(preprocessor)
    result = parser.parse(mol_smiles, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3

    # As we want test CSVFileParser, we assume
    # NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_input_features(dataset[i], expect)

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'Cc1ccccc1'
    assert smiles[2] == 'CC1=CC2CC(CC1)O2'


def test_smiles_parser_target_index(mol_smiles, mols):
    """test `labels` option and retain_smiles=True."""
    preprocessor = NFPPreprocessor()
    parser = SmilesParser(preprocessor)
    result = parser.parse(mol_smiles, return_smiles=True, target_index=[0, 2])
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 2

    # As we want test CSVFileParser, we assume
    # NFPPreprocessor works as documented.
    expect = preprocessor.get_input_features(mols[0])
    check_input_features(dataset[0], expect)

    expect = preprocessor.get_input_features(mols[2])
    check_input_features(dataset[1], expect)

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'CC1=CC2CC(CC1)O2'


def test_smiles_parser_extract_total_num(mol_smiles):
    preprocessor = NFPPreprocessor()
    parser = SmilesParser(preprocessor)
    num = parser.extract_total_num(mol_smiles)
    assert num == 3


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
