from logging import getLogger

import numpy

from chainer import cuda, Variable  # NOQA

from chainer_chemistry.links.scaler.base import BaseScaler, to_array  # NOQA
from chainer_chemistry.links.array.shape_transformer_to_2d import ShapeTransformerTo2D  # NOQA


def format_x(x):
    """x may be array or Variable."""
    # currently, only consider the case x is 2-dim, (batchsize, feature)
    if x.ndim == 1:
        # Deal with as 1 feature with several samples.
        x = x[:, None]
    return x


class MinMaxScaler(BaseScaler):

    def __init__(self):
        super(MinMaxScaler, self).__init__()
        self.indices = None
        self.register_persistent('indices')
        self.min = None
        self.register_persistent('min')
        self.max = None
        self.register_persistent('max')

    def fit(self, x, indices=None, axis=1):
        """Fitting parameter.

        Args:
            x (numpy.ndarray or cupy.ndarray or Variable):
            indices (list or tuple or None):
                indices for applying standard scaling.
            axis (int): axis to calculate min & max.

        Returns:
            self (MinMaxScaler): this instance.
        """
        x = to_array(x)
        x = format_x(x)
        x = ShapeTransformerTo2D(axis=axis).transform(x).array

        if indices is None:
            pass
        elif isinstance(indices, (list, tuple)):
            indices = numpy.asarray(indices)
        self.indices = indices
        if self.indices is not None:
            x = x[:, self.indices]

        xp = self.xp
        if xp is numpy:
            x = cuda.to_cpu(x)
        else:
            x = cuda.to_gpu(x)
        self.min = xp.nanmin(x, axis=0)
        self.max = xp.nanmax(x, axis=0)

        # result consistency check
        if xp.sum(self.max - self.min == 0) > 0:
            logger = getLogger(__name__)
            ind = numpy.argwhere(cuda.to_cpu(self.max-self.min) == 0)[:, 0]
            logger.warning('fit: max-min was 0 at indices {}'.format(ind))
        return self

    def _compute_min_diff_all(self, input_dim):
        diff = self.max - self.min
        diff_nonzero_indices = diff != 0
        if self.indices is None:
            diff_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            diff_all[diff_nonzero_indices] = diff[diff_nonzero_indices]
            return self.min, diff_all
        else:
            min_all = self.xp.zeros(input_dim, dtype=self.xp.float32)
            min_all[self.indices] = self.min
            diff_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            non_zero_indices = self.indices[diff_nonzero_indices]
            diff_all[non_zero_indices] = diff[diff_nonzero_indices]
            return min_all, diff_all

    def transform(self, x, axis=1):
        is_array = not isinstance(x, Variable)
        if self.min is None:
            raise AttributeError(
                '[Error] min is None, call fit beforehand!')
        x = format_x(x)
        shape_transformer = ShapeTransformerTo2D(axis=axis)
        x = shape_transformer.transform(x)
        min_all, diff_all = self._compute_min_diff_all(x.shape[1])
        x = (x - min_all[None, :]) / diff_all[None, :]
        x = shape_transformer.inverse_transform(x)
        if is_array:
            x = x.array
        return x

    def inverse_transform(self, x, axis=1):
        is_array = not isinstance(x, Variable)
        if self.min is None:
            raise AttributeError(
                '[Error] min is None, call fit beforehand!')
        x = format_x(x)
        shape_transformer = ShapeTransformerTo2D(axis=axis)
        x = shape_transformer.transform(x)

        min_all, diff_all = self._compute_min_diff_all(x.shape[1])
        x = x * diff_all[None, :] + min_all[None, :]

        x = shape_transformer.inverse_transform(x)
        if is_array:
            x = x.array
        return x
