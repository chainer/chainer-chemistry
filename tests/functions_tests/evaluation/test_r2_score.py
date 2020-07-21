import numpy
import pytest

from chainer import cuda

import chainer_chemistry


def r2_score(pred, true, sample_weight=None, multioutput="uniform_average",
             ignore_nan=False):
    pred = cuda.to_cpu(pred)
    true = cuda.to_cpu(true)
    diff = pred - true
    dev = true - numpy.mean(true, axis=0)
    if ignore_nan:
        diff[numpy.isnan(diff)] = 0.
        dev[numpy.isnan(dev)] = 0.
    SS_res = numpy.asarray(
        numpy.sum(diff ** 2, axis=0))
    SS_tot = numpy.asarray(
        numpy.sum(dev ** 2, axis=0))

    if multioutput == 'uniform_average':
        if numpy.any(SS_tot == 0):
            return 0.0
        else:
            return (1 - SS_res / SS_tot).mean()
    elif multioutput == 'raw_values':
        if numpy.any(SS_tot == 0):
            # Assign dummy value to avoid zero-division
            SS_tot_iszero = SS_tot == 0
            SS_tot[SS_tot_iszero] = 1

            return numpy.where(SS_tot_iszero, 0.0, 1 - SS_res / SS_tot)
        else:
            return 1 - SS_res / SS_tot


@pytest.fixture
def inputs():
    numpy.random.seed(0)
    x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
    # Add sufficient margin to prevent computational error
    diff = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
    diff[abs(diff) < 0.01] = 0.5
    x1 = x0 + diff
    x2 = numpy.asarray([[0.3, numpy.nan, 0.2],
                        [numpy.nan, 0.1, 0.5],
                        [0.9, 0.7, numpy.nan],
                        [0.2, -0.3, 0.4]]).astype(numpy.float32)
    return x0, x1, x2


def check_forward(inputs):
    x0, x1, _ = inputs
    y = chainer_chemistry.functions.r2_score(x0, x1)
    assert y.data.dtype == 'f'
    assert y.data.shape == ()

    expect = r2_score(x0, x1)
    assert numpy.allclose(cuda.to_cpu(y.data), expect)


def check_forward_ignore_nan(inputs):
    x0, _, x2 = inputs
    y = chainer_chemistry.functions.r2_score(x0, x2, ignore_nan=True)
    assert y.data.dtype == 'f'
    assert y.data.shape == ()

    expect = r2_score(x0, x2, ignore_nan=True)
    assert numpy.allclose(cuda.to_cpu(y.data), expect)


def check_forward_ignore_nan_with_nonnan_value(inputs):
    x0, x1, _ = inputs
    y = chainer_chemistry.functions.r2_score(x0, x1, ignore_nan=True)
    assert y.data.dtype == 'f'
    assert y.data.shape == ()

    expect = r2_score(x0, x1, ignore_nan=True)
    assert numpy.allclose(y.data, expect)


def test_forward_cpu(inputs):
    check_forward(inputs)
    check_forward_ignore_nan(inputs)
    check_forward_ignore_nan_with_nonnan_value(inputs)


@pytest.mark.gpu
def test_forward_gpu(inputs):
    x0, x1, x2 = inputs
    check_forward((cuda.to_gpu(x0), cuda.to_gpu(x1), None))
    check_forward_ignore_nan((cuda.to_gpu(x0), None, cuda.to_gpu(x2)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
