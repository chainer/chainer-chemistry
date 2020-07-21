import numpy
import pytest

from chainer_chemistry.dataset.parsers import SmilesParser
from chainer_chemistry.dataset.preprocessors.weavenet_preprocessor import WeaveNetPreprocessor  # NOQA


@pytest.mark.parametrize('max_atoms', [20, 30])
@pytest.mark.parametrize('use_fixed_atom_feature', [True, False])
def test_weave_preprocessor(max_atoms, use_fixed_atom_feature):
    preprocessor = WeaveNetPreprocessor(
        max_atoms=max_atoms, use_fixed_atom_feature=use_fixed_atom_feature)
    dataset = SmilesParser(preprocessor).parse(
        ['C#N', 'Cc1cnc(C=O)n1C', 'c1ccccc1']
    )["dataset"]

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]
    if use_fixed_atom_feature:
        assert atoms.ndim == 2  # (atom, ch)
        assert atoms.dtype == numpy.float32
    else:
        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
    # (atom from * atom to, ch)
    assert adjs.ndim == 2
    assert adjs.shape[0] == max_atoms * max_atoms
    assert adjs.dtype == numpy.float32

    # TODO(nakago): test feature extraction behavior...
    atoms0, adjs0 = dataset[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
