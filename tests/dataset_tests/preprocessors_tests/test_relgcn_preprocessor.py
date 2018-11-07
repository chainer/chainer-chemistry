import numpy
import pytest

from chainer_chemistry.dataset.parsers import SmilesParser
from chainer_chemistry.dataset.preprocessors import RelGCNPreprocessor


def test_relgcn_preprocessor():
    preprocessor = RelGCNPreprocessor()
    dataset = SmilesParser(preprocessor).parse(
        ['C#N', 'Cc1cnc(C=O)n1C', 'c1ccccc1']
    )["dataset"]

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]
    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (edge_type, atom from, atom to)
    assert adjs.ndim == 3
    assert adjs.dtype == numpy.float32

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


def test_relgcn_preprocessor_kekulize():
    preprocessor = RelGCNPreprocessor(kekulize=True)
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


def test_relgcn_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        pp = RelGCNPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
