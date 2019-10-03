import numpy
import pandas
import pytest

from chainer_chemistry.datasets.mp import get_mp_filepath, filename_list, MPDataset  # NOQA


class DummyPreprocessor:
    def get_input_feature_from_crystal(self, structure):
        return 7, 3


@pytest.mark.slow
@pytest.mark.parametrize('filename', filename_list)
def test_get_mp_filepath(filename):
    filepath = get_mp_filepath(filename)
    print('filepath', filepath)
    assert isinstance(filepath, str)


def test_get_mp_filepath_valueerror():
    with pytest.raises(ValueError):
        get_mp_filepath('hoge')


@pytest.mark.slow
def test_mp_dataset():
    dataset = MPDataset(DummyPreprocessor())
    dataset.get_mp(target_list=['energy'], num_data=10)
    a = dataset[1]
    # print('a', a, type(a[-1]))
    assert a[0] == 7
    assert a[1] == 3
    assert isinstance(a[2], pandas.Series)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
