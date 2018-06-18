import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import chainer_chemistry


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.x2 = numpy.asarray([[0.3, numpy.nan, 0.2],
                                 [numpy.nan, 0.1, 0.5],
                                 [0.9, 0.7, numpy.nan],
                                 [0.2, -0.3, 0.4]]).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)
        self.ggx0 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.ggx1 = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = chainer_chemistry.functions.mean_squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += (self.x0[i] - self.x1[i]) ** 2
        loss_expect /= self.x0.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    def check_forward_ignore_nan(self, x0_data, x2_data):
        x0 = chainer.Variable(x0_data)
        x2 = chainer.Variable(x2_data)
        loss = chainer_chemistry.functions.mean_squared_error(x0, x2,
                                                              ignore_nan=True)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.
        nan_mask = numpy.invert(numpy.isnan(self.x2)).astype(self.x2.dtype)
        for i in numpy.ndindex(self.x0.shape):
            loss_expect += ((self.x0[i] - numpy.nan_to_num(self.x2[i])) ** 2
                            * nan_mask[i])
        loss_expect /= self.x0.size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)
        self.check_forward_ignore_nan(self.x0, self.x2)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data):
        gradient_check.check_backward(
            chainer_chemistry.functions.mean_squared_error,
            (x0_data, x1_data), None, eps=1e-2)

    def check_backward_ignore_nan(self, x0_data, x2_data):
        def func(x0, x1):
            return chainer_chemistry.functions.mean_squared_error(
                x0, x1, ignore_nan=True)
        gradient_check.check_backward(func, (x0_data, x2_data),
                                      None, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1)
        self.check_backward_ignore_nan(self.x0, self.x2)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_double_backward(self, x0_data, x1_data, gy_data,
                              ggx0_data, ggx1_data):

        gradient_check.check_double_backward(
            chainer_chemistry.functions.mean_squared_error, (x0_data, x1_data), # NOQA
            gy_data, (ggx0_data, ggx1_data), eps=1e-2)

    @condition.retry(3)
    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x0, self.x1, self.gy, self.ggx0, self.ggx1)

    @attr.gpu
    @condition.retry(3)
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx0), cuda.to_gpu(self.ggx1))


testing.run_module(__name__, __file__)
