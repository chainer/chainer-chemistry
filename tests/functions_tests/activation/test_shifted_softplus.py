import numpy
import pytest

import chainer
from chainer import cuda
from chainer import gradient_check

import chainer_chemistry
from chainer_chemistry.functions.activation.shifted_softplus import shifted_softplus


def test_forward_cpu():
    x = numpy.array([[1, 2, 3], [6, 5, 4]], dtype=numpy.float32)
    output = shifted_softplus(x)
    expect_output = numpy.array([
        [0.62011445, 1.4337809, 2.3554401],
        [5.3093286, 4.313568, 3.3250027]], dtype=numpy.float32)
    numpy.allclose(output.array, expect_output)


def test_forward_zero_cpu():
    x = numpy.zeros((2, 3), dtype=numpy.float32)
    output = shifted_softplus(x)
    expect_output = numpy.zeros((2, 3), dtype=numpy.float32)
    numpy.allclose(output.array, expect_output)


def test_forward_avoid_overflow_cpu():
    x = numpy.array([1e5], dtype=numpy.float32)
    output = shifted_softplus(x)
    expect_output = numpy.array([1e5], dtype=numpy.float32)
    numpy.allclose(output.array, expect_output)


@pytest.mark.gpu
def test_forward_gpu():
    x = cuda.to_gpu(numpy.array([[1, 2, 3], [6, 5, 4]], dtype=numpy.float32))
    output = shifted_softplus(x)
    expect_output = numpy.array([
        [0.62011445, 1.4337809, 2.3554401],
        [5.3093286, 4.313568, 3.3250027]], dtype=numpy.float32)
    numpy.allclose(cuda.to_cpu(output.array), expect_output)


@pytest.mark.gpu
def test_forward_zero_gpu():
    x = cuda.to_gpu(numpy.zeros((2, 3), dtype=numpy.float32))
    output = shifted_softplus(x)
    expect_output = numpy.zeros((2, 3), dtype=numpy.float32)
    numpy.allclose(cuda.to_cpu(output.array), expect_output)


@pytest.mark.gpu
def test_forward_avoid_overflow_gpu():
    x = cuda.to_gpu(numpy.array([1e5], dtype=numpy.float32))
    output = shifted_softplus(x)
    expect_output = numpy.array([1e5], dtype=numpy.float32)
    numpy.allclose(cuda.to_cpu(output.array), expect_output)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
