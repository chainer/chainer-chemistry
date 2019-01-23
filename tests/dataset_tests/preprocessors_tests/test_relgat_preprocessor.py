import numpy
import pytest

from chainer_chemistry.dataset.parsers.smiles_parser import SmilesParser
from chainer_chemistry.dataset.preprocessors.relgat_preprocessor import RelGATPreprocessor  # NOQA


# All tests are copied from GGNNPreprocessor now.
@pytest.mark.parametrize('return_is_real_node', [True, False])
def test_relgat_preprocessor(return_is_real_node):
    preprocessor = RelGATPreprocessor(return_is_real_node=return_is_real_node)
    dataset = SmilesParser(preprocessor).parse(
        ['C#N', 'Cc1cnc(C=O)n1C', 'c1ccccc1']
    )["dataset"]

    index = numpy.random.choice(len(dataset), None)
    if return_is_real_node:
        atoms, adjs, is_real_node = dataset[index]
        assert is_real_node.ndim == 1  # (atom,)
        assert is_real_node.dtype == numpy.float32  # (atom,)
    else:
        atoms, adjs = dataset[index]
    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (edge_type, atom from, atom to)
    assert adjs.ndim == 3
    assert adjs.dtype == numpy.float32

    if return_is_real_node:
        atoms0, adjs0, is_real_node0 = dataset[0]
    else:
        atoms0, adjs0 = dataset[0]
    assert numpy.allclose(atoms0, numpy.array([6, 7], numpy.int32))
    expect_adjs = numpy.array(
        [[[0., 0.],
          [0., 0.]],
         [[0., 0.],
          [0., 0.]],
         [[0., 1.],
          [1., 0.]],
         [[0., 0.],
          [0., 0.]]], dtype=numpy.float32)
    assert numpy.allclose(adjs0, expect_adjs)
    if return_is_real_node:
        expect_is_real_node = numpy.array([1, 1], numpy.float32)
        assert numpy.allclose(is_real_node0, expect_is_real_node)

    if return_is_real_node:
        atoms1, adjs1, is_real_node1 = dataset[1]
    else:
        atoms1, adjs1 = dataset[1]
    assert numpy.allclose(
        atoms1, numpy.array([6, 6, 6, 7, 6, 6, 8, 7, 6], numpy.int32))
    # include aromatic bond (ch=3)
    expect_adjs = numpy.array(
        [[[0., 1., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 1., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 1., 0.],
          [0., 1., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=numpy.float32)
    assert numpy.allclose(adjs1, expect_adjs)
    if return_is_real_node:
        expect_is_real_node = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1],
                                          numpy.float32)
        assert numpy.allclose(is_real_node1, expect_is_real_node)


def test_relgat_preprocessor_kekulize():
    preprocessor = RelGATPreprocessor(kekulize=True, return_is_real_node=False)
    dataset = SmilesParser(preprocessor).parse(
        ['C#N', 'Cc1cnc(C=O)n1C', 'c1ccccc1']
    )["dataset"]
    atoms1, adjs1 = dataset[1]
    assert numpy.allclose(
        atoms1, numpy.array([6, 6, 6, 7, 6, 6, 8, 7, 6], numpy.int32))
    # NOT include aromatic bond (ch=3)
    expect_adjs = numpy.array(
        [[[0., 1., 0., 0., 0., 0., 0., 0., 0.],
          [1., 0., 0., 0., 0., 0., 0., 1., 0.],
          [0., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 1., 0.],
          [0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 1., 0., 0., 0., 1.],
          [0., 0., 0., 0., 0., 0., 0., 1., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 1., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],
         [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=numpy.float32)
    assert numpy.allclose(adjs1, expect_adjs)


def test_relgat_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        pp = RelGATPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
