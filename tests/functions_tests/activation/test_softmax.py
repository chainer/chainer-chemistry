import numpy
import pytest

from chainer import cuda

from chainer_chemistry.functions.activation.softmax import softmax


def test_forward_cpu():
    x = numpy.array([[1, 2, 3], [6, 5, 4]], dtype=numpy.float32)
    output = softmax(x)
    expect_output = numpy.array([
        [0.09003057, 0.24472848, 0.66524094],
        [0.66524094, 0.24472848, 0.09003057]], dtype=numpy.float32)
    numpy.allclose(output.array, expect_output)


def test_forward_cpu_with_mask():
    x = numpy.array([[1, 2, 3, 2, 5], [1, 6, 5, 4, 2]], dtype=numpy.float32)
    mask = numpy.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]], dtype=numpy.float32)
    output = softmax(x, mask=mask)
    expect_output = numpy.array([
        [0.09003057, 0.24472848, 0.66524094, 0., 0.],
        [0., 0.66524094, 0.24472848, 0.09003057, 0.]], dtype=numpy.float32)
    numpy.allclose(output.array, expect_output)


@pytest.mark.gpu
def test_forward_gpu():
    x = cuda.to_gpu(numpy.array([[1, 2, 3], [6, 5, 4]], dtype=numpy.float32))
    output = softmax(x)
    expect_output = numpy.array([
        [0.09003057, 0.24472848, 0.66524094],
        [0.66524094, 0.24472848, 0.09003057]], dtype=numpy.float32)
    numpy.allclose(cuda.to_cpu(output.array), expect_output)


@pytest.mark.gpu
def test_forward_gpu_with_mask():
    x = numpy.array([[1, 2, 3, 2, 5], [1, 6, 5, 4, 2]], dtype=numpy.float32)
    mask = numpy.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]], dtype=numpy.float32)
    x, mask = map(cuda.to_gpu, (x, mask))
    output = softmax(x, mask=mask)
    expect_output = numpy.array([
        [0.09003057, 0.24472848, 0.66524094, 0., 0.],
        [0., 0.66524094, 0.24472848, 0.09003057, 0.]], dtype=numpy.float32)
    numpy.allclose(cuda.to_cpu(output.array), expect_output)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
