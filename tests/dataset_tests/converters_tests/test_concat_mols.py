import numpy
import pytest

from chainerchem.dataset.converters import concat_mols


@pytest.fixture
def data():
    a = numpy.array([1, 2], dtype=numpy.float32)
    b = numpy.array([3, 4, 5], dtype=numpy.float32)
    c = numpy.array([6, 7], dtype=numpy.float32)
    return a, b, c


def test_concat_mols(data):
    actual = concat_mols(data)
    expect = numpy.array([[1, 2, 0],
                          [3, 4, 5],
                          [6, 7, 0]],
                         dtype=numpy.float32)
    numpy.testing.assert_array_equal(actual, expect)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
