import chainer
import numpy
import pytest

from chainer_chemistry.dataset.converters import concat_mols


@pytest.fixture
def data_1d():
    a = numpy.array([1, 2])
    b = numpy.array([4, 5, 6])
    return a, b


@pytest.fixture
def data_1d_expect():
    a = numpy.array([1, 2, 0])
    b = numpy.array([4, 5, 6])
    return a, b


@pytest.fixture
def data_2d():
    a = numpy.array([[1, 2], [3, 4]])
    b = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return a, b


@pytest.fixture
def data_2d_expect():
    a = numpy.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
    b = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return a, b


def test_concat_mols_1d_cpu(data_1d, data_1d_expect):
    result = concat_mols(data_1d, device=-1)
    assert numpy.array_equal(result[0], data_1d_expect[0])
    assert numpy.array_equal(result[1], data_1d_expect[1])


def test_concat_mols_2d_cpu(data_2d, data_2d_expect):
    result = concat_mols(data_2d, device=-1)
    assert numpy.array_equal(result[0], data_2d_expect[0])
    assert numpy.array_equal(result[1], data_2d_expect[1])


@pytest.mark.gpu
def test_concat_mols_1d_gpu(data_1d, data_1d_expect):
    result = concat_mols(data_1d, device=0)
    assert chainer.cuda.get_device_from_array(result[0]).id == 0
    assert chainer.cuda.get_device_from_array(result[1]).id == 0
    assert numpy.array_equal(chainer.cuda.to_cpu(result[0]),
                             data_1d_expect[0])
    assert numpy.array_equal(chainer.cuda.to_cpu(result[1]),
                             data_1d_expect[1])


@pytest.mark.gpu
def test_concat_mols_2d_gpu(data_2d, data_2d_expect):
    result = concat_mols(data_2d, device=0)
    assert chainer.cuda.get_device_from_array(result[0]).id == 0
    assert chainer.cuda.get_device_from_array(result[1]).id == 0
    assert numpy.array_equal(chainer.cuda.to_cpu(result[0]),
                             data_2d_expect[0])
    assert numpy.array_equal(chainer.cuda.to_cpu(result[1]),
                             data_2d_expect[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
