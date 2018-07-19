import numpy

from chainer import cuda
from chainer import function_node
import chainer.functions
from chainer.utils import type_check


class MeanSquaredError(function_node.FunctionNode):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def __init__(self, ignore_nan=False):
        # TODO(mottodora): implement task weight calculation
        self.ignore_nan = ignore_nan

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        # TODO(mottodora): add reduce option
        if self.ignore_nan:
            diff[numpy.isnan(diff)] = 0.
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        # TODO(mottodora): add reduce option
        if self.ignore_nan:
            diff[cupy.isnan(diff)] = 0.
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, indexes, gy):
        x0, x1 = self.get_retained_inputs()
        xp = cuda.get_array_module(x0)
        ret = []
        diff = x0 - x1
        if self.ignore_nan:
            diff = chainer.functions.where(xp.isnan(diff.array),
                                           xp.zeros_like(diff.array), diff)
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2. / diff.size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret


def mean_squared_error(x0, x1, ignore_nan=False):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    Args:
        x0 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        ignore_nan (bool): If `True`, this function compute mean squared error
            ignoring NaNs. The arithmetic mean is the sum of the non-NaN
            elements along the axis divided by the number of whole elements.

    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean squared
            error of two inputs.
    """
    return MeanSquaredError(ignore_nan).apply((x0, x1))[0]
