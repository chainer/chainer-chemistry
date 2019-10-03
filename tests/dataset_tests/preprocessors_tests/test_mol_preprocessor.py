import pytest
from rdkit import Chem

from chainer_chemistry.dataset.preprocessors import MolPreprocessor


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    ret.SetProp('foo', '1')
    ret.SetProp('bar', '2')
    return ret


@pytest.fixture
def pp():
    return MolPreprocessor()


class TestGetLabel(object):

    def test_default(self, mol, pp):
        labels = pp.get_label(mol)
        assert labels == []

    def test_empty(self, mol, pp):
        labels = pp.get_label(mol, [])
        assert labels == []

    def test_one_label(self, mol, pp):
        labels = pp.get_label(mol, ['foo'])
        assert labels == ['1']

    def test_two_labels(self, mol, pp):
        labels = pp.get_label(mol, ['bar', 'foo'])
        assert labels == ['2', '1']

    def test_non_existent_label(self, mol, pp):
        labels = pp.get_label(mol, ['foo', 'buz'])
        assert labels == ['1', None]


if __name__ == '__main__':
    pytest.main()
