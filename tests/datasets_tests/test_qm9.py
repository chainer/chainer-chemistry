import os
import pytest

from chainer_chemistry.datasets import qm9


@pytest.mark.slow
def test_get_qm9_filepath_without_download():
    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    filepath = qm9.get_qm9_filepath(dataset_type, download_if_not_exist=False)
    assert isinstance(filepath, str)
    assert not os.path.exists(filepath)


@pytest.mark.slow
def test_get_qm9_filepath_with_download():
    filepath = qm9.get_qm9_filepath(download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)  # ensure a cache file does not exist.

    # This method downloads the file if not exist
    filepath = qm9.get_qm9_filepath(download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)
    os.remove(filepath)


# TODO
# def test_get_qm9():
#     train, val, test = qm9.get_qm9(preprocess_method='nfp')
#
#     # Test only NFP preprocessing here...
#     for dataset in [train, val, test]:
#         index = numpy.random.choice(len(dataset), None)
#         atoms, adjs, label = dataset[index]
#
#         assert atoms.ndim == 1  # (atom, )
#         assert atoms.dtype == numpy.int32
#         # (atom from, atom to) or (edge_type, atom from, atom to)
#         assert adjs.ndim == 2 or adjs.ndim == 3
#         assert adjs.dtype == numpy.float32
#         assert label.ndim == 1
#         assert label.dtype == numpy.int32


def test_get_qm9_label_names():
    label_names = qm9.get_qm9_label_names()
    assert isinstance(label_names, list)
    for label in label_names:
        assert isinstance(label, str)


if __name__ == '__main__':
    args = ['-s', ]
    pytest.main(args=args)
