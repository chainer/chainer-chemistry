import numpy
import pytest

from chainer_chemistry.dataset.parsers import SDFFileParser
from chainer_chemistry.dataset.preprocessors import RelGATPreprocessor
from chainer_chemistry.datasets import get_tox21_filepath


@pytest.mark.slow
def test_gat_preprocessor():
    preprocessor = RelGATPreprocessor()

    def postprocess_label(label_list):
        # Set -1 to the place where the label is not found,
        # this corresponds to not calculate loss with `sigmoid_cross_entropy`
        return [-1 if label is None else label for label in label_list]

    dataset = SDFFileParser(preprocessor, postprocess_label=postprocess_label
                            ).parse(get_tox21_filepath('train'))["dataset"]

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (edge_type, atom from, atom to)
    assert adjs.ndim == 3
    assert adjs.dtype == numpy.float32


def test_gat_preprocessor_assert_raises():
    with pytest.raises(ValueError):
        pp = RelGATPreprocessor(max_atoms=3, out_size=2)  # NOQA


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
