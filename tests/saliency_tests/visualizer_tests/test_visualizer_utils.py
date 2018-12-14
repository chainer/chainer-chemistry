import numpy
import pytest

from chainer_chemistry.saliency.visualizer.visualizer_utils import abs_max_scaler  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import min_max_scaler  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import normalize_scaler  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import red_blue_cmap  # NOQA


def test_abs_max_scaler():
    saliency = numpy.array([1., 2., 3.])
    result = abs_max_scaler(saliency)
    expected = numpy.array([1. / 3, 2. / 3., 1.])
    assert numpy.allclose(result, expected)

    # test with 0 arrays
    saliency = numpy.array([0, 0, 0])
    result = abs_max_scaler(saliency)
    expected = numpy.array([0, 0, 0])
    assert numpy.allclose(result, expected)


def test_min_max_scaler():
    saliency = numpy.array([1., -3., 3.])
    result = min_max_scaler(saliency)
    expected = numpy.array([4. / 6, 0., 1.])
    assert numpy.allclose(result, expected)

    # test with 0 arrays
    saliency = numpy.array([0, 0, 0])
    result = min_max_scaler(saliency)
    expected = numpy.array([0, 0, 0])
    assert numpy.allclose(result, expected)


def test_normalize_scaler():
    saliency = numpy.array([1., 2., 3.])
    result = normalize_scaler(saliency)
    expected = numpy.array([1./6., 2./6, 3./6.])
    assert numpy.allclose(result, expected)

    # test with 0 arrays
    saliency = numpy.array([0, 0, 0])
    result = normalize_scaler(saliency)
    expected = numpy.array([0, 0, 0])
    assert numpy.allclose(result, expected)


def test_red_blue_cmap():
    assert red_blue_cmap(1) == (1., 0., 0.)  # Red
    assert red_blue_cmap(0) == (1., 1., 1.)  # White
    assert red_blue_cmap(-1) == (0., 0., 1.)  # Blue


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
