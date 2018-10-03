import os

from chainer_chemistry.datasets.molnet import pdbbind_time


def test_get_pdbbind_time_filepath():
    filepath = pdbbind_time.get_pdbbind_time_filepath(
        download_if_not_exist=False)
    if os.path.exists(filepath):
        os.remove(filepath)

    filepath = pdbbind_time.get_pdbbind_time_filepath(
        download_if_not_exist=True)
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)


def test_get_pdbbind_time():
    time_list = pdbbind_time.get_pdbbind_time()
    assert isinstance(time_list, list)
    for time in time_list:
        assert 1900 < time < 2100
