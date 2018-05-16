import os

import numpy
import pytest

from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.datasets import tox21


TOX21_NUM_LABEL = 12
dataset_types = [
    'train',
    'val',
    'test'
]


@pytest.mark.parametrize('dataset_type', dataset_types)
def test_get_tox21_filepath_without_download(dataset_type):
    filepath = tox21.get_tox21_filepath(dataset_type,
                                        download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = tox21.get_tox21_filepath(dataset_type,
                                        download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)


@pytest.mark.slow
@pytest.mark.parametrize('dataset_type', dataset_types)
def test_get_tox21_filepath_with_download(dataset_type):
    filepath = tox21.get_tox21_filepath(dataset_type,
                                        download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    # This method downloads the file if not exist
    filepath = tox21.get_tox21_filepath(dataset_type,
                                        download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)


@pytest.mark.slow
def test_get_tox21():
    # test default behavior
    pp = AtomicNumberPreprocessor()
    train, val, test = tox21.get_tox21(preprocessor=pp)

    # --- Test dataset is correctly obtained ---
    for dataset in [train, val, test]:
        index = numpy.random.choice(len(dataset), None)
        atoms, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        assert label.ndim == 1
        assert label.shape[0] == TOX21_NUM_LABEL
        assert label.dtype == numpy.int32


def test_get_tox21_label_names():
    label_names = tox21.get_tox21_label_names()
    assert isinstance(label_names, list)
    for label in label_names:
        assert isinstance(label, str)


def test_get_tox21_filepath_assert_raises():
    with pytest.raises(ValueError):
        tox21.get_tox21_filepath('other')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
