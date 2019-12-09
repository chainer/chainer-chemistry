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


class MaxAbsScaler(BaseScaler):

    def __init__(self):
        super(MaxAbsScaler, self).__init__()
        self.indices = None
        self.register_persistent('indices')
        self.max_abs = None
        self.register_persistent('max_abs')

    def fit(self, x, indices=None, axis=1):
        """Fitting parameter.

        Args:
            x (numpy.ndarray or cupy.ndarray or Variable):
            indices (list or tuple or None):
                indices for applying standard scaling.
            axis (int): axis to calculate mean & std.

        Returns:
            self (StandardScaler): this instance.
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
        self.max_abs = xp.nanmax(xp.abs(x), axis=0)

        # result consistency check
        if xp.sum(self.max_abs == 0) > 0:
            logger = getLogger(__name__)
            ind = numpy.argwhere(cuda.to_cpu(self.max_abs) == 0)[:, 0]
            logger.warning('fit: max_abs was 0 at indices {}'.format(ind))
        return self

    def _compute_max_abs_all(self, input_dim):
        if self.indices is None:
            max_abs_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            max_abs_all[self.max_abs != 0] = self.max_abs[self.max_abs != 0]
            return max_abs_all
        else:
            max_abs_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            non_zero_indices = self.indices[self.max_abs != 0]
            max_abs_all[non_zero_indices] = self.max_abs[self.max_abs != 0]
            return max_abs_all

    def transform(self, x, axis=1):
        is_array = not isinstance(x, Variable)
        if self.max_abs is None:
            raise AttributeError(
                '[Error] max_abs is None, call fit beforehand!')
        x = format_x(x)
        shape_transformer = ShapeTransformerTo2D(axis=axis)
        x = shape_transformer.transform(x)
        max_abs_all = self._compute_max_abs_all(x.shape[1])
        x = x / max_abs_all[None, :]
        x = shape_transformer.inverse_transform(x)
        if is_array:
            x = x.array
        return x

    def inverse_transform(self, x, axis=1):
        is_array = not isinstance(x, Variable)
        if self.max_abs is None:
            raise AttributeError(
                '[Error] max_abs is None, call fit beforehand!')
        x = format_x(x)
        shape_transformer = ShapeTransformerTo2D(axis=axis)
        x = shape_transformer.transform(x)
        max_abs_all = self._compute_max_abs_all(x.shape[1])
        x = x * max_abs_all[None, :]
        x = shape_transformer.inverse_transform(x)
        if is_array:
            x = x.array
        return x
