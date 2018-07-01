import numpy
import pytest

import chainer
from chainer import cuda
from chainer import gradient_check

import chainer_chemistry


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


@pytest.fixture
def grads():
    numpy.random.seed(0)
    gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
    ggx0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
    ggx1 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
    return gy, ggx0, ggx1


def check_forward(inputs):
    x0_data, x1_data, _ = inputs
    x0 = chainer.Variable(x0_data)
    x1 = chainer.Variable(x1_data)
    loss = chainer_chemistry.functions.mean_absolute_error(x0, x1)
    loss_value = cuda.to_cpu(loss.data)
    assert loss.dtype == numpy.float32
    assert loss_value.shape == ()

    loss_expect = numpy.zeros(())
    x0_data = cuda.to_cpu(x0_data)
    x1_data = cuda.to_cpu(x1_data)
    for i in numpy.ndindex(x0_data.shape):
        loss_expect += abs((x0_data[i] - x1_data[i]))
    loss_expect /= x0_data.size
    assert numpy.allclose(loss_value, loss_expect)


def check_forward_ignore_nan(inputs):
    x0_data, _, x2_data = inputs
    x0 = chainer.Variable(x0_data)
    x2 = chainer.Variable(x2_data)
    loss = chainer_chemistry.functions.mean_absolute_error(x0, x2,
                                                           ignore_nan=True)
    loss_value = cuda.to_cpu(loss.data)
    assert loss.dtype == numpy.float32
    assert loss_value.shape == ()

    loss_expect = numpy.zeros(())
    x0_data = cuda.to_cpu(x0_data)
    x2_data = cuda.to_cpu(x2_data)
    nan_mask = numpy.invert(numpy.isnan(x2_data)).astype(x2_data.dtype)
    for i in numpy.ndindex(x0_data.shape):
        loss_expect += abs(x0_data[i] -
                           numpy.nan_to_num(x2_data[i])) * nan_mask[i]
    loss_expect /= x0_data.size
    assert numpy.allclose(loss_value, loss_expect)


def check_forward_ignore_nan_with_nonnan_value(inputs):
    x0_data, x1_data, _ = inputs
    x0 = chainer.Variable(x0_data)
    x1 = chainer.Variable(x1_data)
    loss = chainer_chemistry.functions.mean_absolute_error(x0, x1,
                                                           ignore_nan=True)
    loss_value = cuda.to_cpu(loss.data)
    assert loss.dtype == numpy.float32
    assert loss_value.shape == ()

    loss_expect = numpy.zeros(())
    x0_data = cuda.to_cpu(x0_data)
    x1_data = cuda.to_cpu(x1_data)
    nan_mask = numpy.invert(numpy.isnan(x1_data)).astype(x1_data.dtype)
    for i in numpy.ndindex(x0_data.shape):
        loss_expect += abs(x0_data[i] -
                           numpy.nan_to_num(x1_data[i])) * nan_mask[i]
    loss_expect /= x0_data.size
    assert numpy.allclose(loss_value, loss_expect)


def test_forward_cpu(inputs):
    check_forward(inputs)
    check_forward_ignore_nan(inputs)
    check_forward_ignore_nan_with_nonnan_value(inputs)


@pytest.mark.gpu
def test_forward_gpu(inputs):
    x0, x1, x2 = inputs
    check_forward((cuda.to_gpu(x0), cuda.to_gpu(x1), None))
    check_forward_ignore_nan((cuda.to_gpu(x0), None, cuda.to_gpu(x2)))


def check_backward(inputs):
    x0_data, x1_data, _ = inputs
    gradient_check.check_backward(
        chainer_chemistry.functions.mean_absolute_error,
        (x0_data, x1_data), None, eps=1e-2)


def check_backward_ignore_nan(inputs):
    x0_data, _, x2_data = inputs

    def func(x0, x1):
        return chainer_chemistry.functions.mean_absolute_error(x0, x1,
                                                               ignore_nan=True)
    gradient_check.check_backward(func, (x0_data, x2_data), None, eps=1e-2)


def check_backward_ignore_nan_with_nonnan_value(inputs):
    x0_data, x1_data, _ = inputs

    def func(x0, x1):
        return chainer_chemistry.functions.mean_absolute_error(x0, x1,
                                                               ignore_nan=True)
    gradient_check.check_backward(func, (x0_data, x1_data), None, eps=1e-2)


def test_backward_cpu(inputs):
    check_backward(inputs)
    check_backward_ignore_nan(inputs)
    check_backward_ignore_nan_with_nonnan_value(inputs)


@pytest.mark.gpu
def test_backward_gpu(inputs):
    x0, x1, x2 = inputs
    check_backward((cuda.to_gpu(x0), cuda.to_gpu(x1), None))
    check_backward_ignore_nan((cuda.to_gpu(x0), None, cuda.to_gpu(x2)))
    check_backward_ignore_nan_with_nonnan_value((cuda.to_gpu(x0),
                                                 cuda.to_gpu(x1), None))


def check_double_backward(inputs, grads):
    x0, x1, _ = inputs
    gy, ggx0, ggx1 = grads

    def func(*xs):
        y = chainer_chemistry.functions.mean_absolute_error(*xs)
        return y * y
    gradient_check.check_double_backward(func, (x0, x1), gy, (ggx0, ggx1))


def check_double_backward_ignore_nan(inputs, grads):
    x0, _, x2 = inputs
    gy, ggx0, ggx1 = grads

    def func(*xs):
        y = chainer_chemistry.functions.mean_absolute_error(*xs,
                                                            ignore_nan=True)
        return y * y
    gradient_check.check_double_backward(func, (x0, x2), gy, (ggx0, ggx1))


def check_double_backward_ignore_nan_with_nonnan_value(inputs, grads):
    x0, x1, _ = inputs
    gy, ggx0, ggx1 = grads

    def func(*xs):
        y = chainer_chemistry.functions.mean_absolute_error(*xs,
                                                            ignore_nan=True)
        return y * y
    gradient_check.check_double_backward(func, (x0, x1), gy, (ggx0, ggx1))


def test_double_backward_cpu(inputs, grads):
    check_double_backward(inputs, grads)
    check_double_backward_ignore_nan(inputs, grads)
    check_double_backward_ignore_nan_with_nonnan_value(inputs, grads)


@pytest.mark.gpu
def test_double_backward_gpu(inputs, grads):
    x0, x1, x2 = inputs
    gy, ggx0, ggx1 = grads
    check_double_backward((cuda.to_gpu(x0), cuda.to_gpu(x1), None),
                          (cuda.to_gpu(gy), cuda.to_gpu(ggx0),
                           cuda.to_gpu(ggx1)))
    check_double_backward_ignore_nan_with_nonnan_value((cuda.to_gpu(x0),
                                                        cuda.to_gpu(x1),
                                                        None),
                                                       (cuda.to_gpu(gy),
                                                        cuda.to_gpu(ggx0),
                                                        cuda.to_gpu(ggx1)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
