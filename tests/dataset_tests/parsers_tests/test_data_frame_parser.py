import numpy
import pandas
import pytest
from rdkit import Chem
import six

from chainer_chemistry.dataset.parsers import DataFrameParser
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


@pytest.fixture()
def label_a():
    return [2.1, 5.3, -1.2]


@pytest.fixture()
def data_frame(mol_smiles, label_a):
    df = pandas.DataFrame({
        'smiles': mol_smiles,
        'labelA': label_a
    })
    return df


def check_input_features(actual, expect):
    assert len(actual) == len(expect)
    for d, e in six.moves.zip(actual, expect):
        numpy.testing.assert_array_equal(d, e)


def check_features(actual, expect_input_features, expect_label):
    assert len(actual) == len(expect_input_features) + 1
    # input features testing
    for d, e in six.moves.zip(actual[:-1], expect_input_features):
        numpy.testing.assert_array_equal(d, e)
    # label testing
    assert actual[-1] == expect_label


def test_data_frame_parser_not_return_smiles(data_frame, mols):
    """Test default behavior"""
    preprocessor = NFPPreprocessor()
    parser = DataFrameParser(preprocessor, smiles_col='smiles')
    # Actually, `dataset, smiles = parser.parse(..)` is enough.
    result = parser.parse(data_frame, return_smiles=False)
    dataset = result['dataset']
    smiles = result['smiles']
    success_flag = result['success_flag']
    assert len(dataset) == 3
    assert smiles is None
    assert success_flag is None

    # As we want test DataFrameParser, we assume
    # NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_input_features(dataset[i], expect)


def test_data_frame_parser_return_smiles(data_frame, mols, label_a):
    """test `labels` option and retain_smiles=True."""
    preprocessor = NFPPreprocessor()
    parser = DataFrameParser(preprocessor, labels='labelA',
                             smiles_col='smiles')
    result = parser.parse(data_frame, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 3

    # We assume NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_features(dataset[i], expect, label_a[i])

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'Cc1ccccc1'
    assert smiles[2] == 'CC1=CC2CC(CC1)O2'


def test_data_frame_parser_target_index(data_frame, mols, label_a):
    """test `labels` option and retain_smiles=True."""
    preprocessor = NFPPreprocessor()
    parser = DataFrameParser(preprocessor, labels='labelA',
                             smiles_col='smiles')
    result = parser.parse(data_frame, return_smiles=True, target_index=[0, 2],
                          return_success_flag=True)
    dataset = result['dataset']
    smiles = result['smiles']
    assert len(dataset) == 2
    success_flag = result['success_flag']
    assert numpy.alltrue(success_flag)
    assert len(success_flag) == 2

    # We assume NFPPreprocessor works as documented.
    expect = preprocessor.get_input_features(mols[0])
    check_features(dataset[0], expect, label_a[0])

    expect = preprocessor.get_input_features(mols[2])
    check_features(dataset[1], expect, label_a[2])

    # check smiles array
    assert type(smiles) == numpy.ndarray
    assert smiles.ndim == 1
    assert len(smiles) == len(dataset)
    assert smiles[0] == 'CN=C=O'
    assert smiles[1] == 'CC1=CC2CC(CC1)O2'


def test_data_frame_parser_return_success_flag(mols, label_a):
    """test `labels` option and retain_smiles=True."""
    preprocessor = NFPPreprocessor()
    parser = DataFrameParser(preprocessor, labels='labelA',
                             smiles_col='smiles')
    df = pandas.DataFrame({
        'smiles': ['var', 'CN=C=O', 'hoge', 'Cc1ccccc1', 'CC1=CC2CC(CC1)O2'],
        'labelA': [0., 2.1, 0., 5.3, -1.2],
    })
    result = parser.parse(df, return_smiles=True, return_success_flag=True)

    dataset = result['dataset']
    # smiles = result['smiles']
    assert len(dataset) == 3
    success_flag = result['success_flag']
    assert len(success_flag) == 5
    # print('success_flag', success_flag)
    assert numpy.alltrue(success_flag[[1, 3, 4]])
    assert numpy.alltrue(~success_flag[[0, 2]])

    # We assume NFPPreprocessor works as documented.
    for i in range(3):
        expect = preprocessor.get_input_features(mols[i])
        check_features(dataset[i], expect, label_a[i])


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
