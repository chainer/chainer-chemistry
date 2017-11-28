import os
import numpy
import pytest

from chainerchem.dataset.parsers import SDFFileParser
from chainerchem.dataset.preprocessors import GGNNPreprocessor
from chainerchem.datasets import get_tox21_filepath


@pytest.mark.slow
def test_ggnn_preprocessor():
    preprocessor = GGNNPreprocessor()

    def postprocess_label(label_list):
        # Set -1 to the place where the label is not found,
        # this corresponds to not calculate loss with `sigmoid_cross_entropy`
        return [-1 if label is None else label for label in label_list]

    dataset = SDFFileParser(preprocessor, postprocess_label=postprocess_label
                            ).parse(get_tox21_filepath('train'))

    index = numpy.random.choice(len(dataset), None)
    atoms, adjs = dataset[index]

    assert atoms.ndim == 1  # (atom, )
    assert atoms.dtype == numpy.int32
    # (edge_type, atom from, atom to)
    assert adjs.ndim == 3
    assert adjs.dtype == numpy.float32

if __name__ == '__main__':
    pytest.main()
