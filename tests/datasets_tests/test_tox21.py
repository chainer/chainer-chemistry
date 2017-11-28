import os
import numpy
import pytest

from chainerchem.datasets import tox21


dataset_types = [
    'train',
    'val',
    'test'
]


@pytest.mark.parametrize('dataset_type', dataset_types)
def test_get_tox21_filepath_without_download(dataset_type):
    filepath = tox21.get_tox21_filepath(dataset_type, download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = tox21.get_tox21_filepath(dataset_type, download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)


@pytest.mark.slow
@pytest.mark.parametrize('dataset_type', dataset_types)
def test_get_tox21_filepath_with_download(dataset_type):
    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    # This method downloads the file if not exist
    filepath = tox21.get_tox21_filepath(dataset_type, download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)
    os.remove(filepath)


@pytest.mark.slow
def test_get_tox21():
    train, val, test = tox21.get_tox21(preprocess_method='nfp')

    # Test only NFP preprocessing here...
    for dataset in [train, val, test]:
        index = numpy.random.choice(len(dataset), None)
        atoms, adjs, label = dataset[index]

        assert atoms.ndim == 1  # (atom, )
        assert atoms.dtype == numpy.int32
        # (atom from, atom to) or (edge_type, atom from, atom to)
        assert adjs.ndim == 2 or adjs.ndim == 3
        assert adjs.dtype == numpy.float32
        assert label.ndim == 1
        assert label.dtype == numpy.int32


def test_get_tox21_label_names():
    label_names = tox21.get_tox21_label_names()
    assert isinstance(label_names, list)
    for label in label_names:
        assert isinstance(label, str)


if __name__ == '__main__':
    pytest.main()
