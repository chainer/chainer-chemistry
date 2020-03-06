import pytest
from rdkit import Chem

from chainer_chemistry.dataset.preprocessors.gwm_preprocessor import (
    NFPGWMPreprocessor, GGNNGWMPreprocessor, GINGWMPreprocessor,
    RSGCNGWMPreprocessor)  # NOQA


@pytest.fixture
def mol():
    ret = Chem.MolFromSmiles('CN=C=O')
    return ret


@pytest.mark.parametrize('pp_type', [
    NFPGWMPreprocessor, GGNNGWMPreprocessor, GINGWMPreprocessor,
    RSGCNGWMPreprocessor])
def test_gwm_preprocessor(mol, pp_type):
    pp = pp_type()
    ret = pp.get_input_features(mol)
    # currently all preprocessor returns `super_node_x` at 3rd args.
    assert len(ret) == 3
    super_node_x = ret[2]

    # print('super_node_x', super_node_x.shape, super_node_x)
    assert super_node_x.ndim == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
