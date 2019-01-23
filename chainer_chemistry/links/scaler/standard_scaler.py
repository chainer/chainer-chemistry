import numpy

from chainer_chemistry.links.scaler.base import BaseScaler, to_array  # NOQA


def format_x(x):
    """x may be array or Variable"""
    # currently, only consider the case x is 2-dim, (batchsize, feature)
    if x.ndim == 1:
        # Deal with as 1 feature with several samples.
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(
            "Unexpected value x.shape={}, only x.ndim=2 is supported."
            .format(x.shape))
    return x


class StandardScaler(BaseScaler):

    def __init__(self):
        super(StandardScaler, self).__init__()
        self.indices = None
        self.register_persistent('indices')
        self.mean = None
        self.register_persistent('mean')
        self.std = None
        self.register_persistent('std')

    def fit(self, x, indices=None):
        """Fitting parameter.

        Args:
            x:
            indices (list or tuple or None):
                indices for applying standard scaling.

        Returns:
            self (StandardScaler): this instance.
        """
        x = to_array(x)
        x = format_x(x)

        if indices is None:
            pass
        elif isinstance(indices, (list, tuple)):
            indices = numpy.asarray(indices)
        self.indices = indices
        if self.indices is not None:
            x = x[:, self.indices]
        self.mean = self.xp.nanmean(x, axis=0)
        self.std = self.xp.nanstd(x, axis=0)
        return self

    def _compute_mean_std_all(self, input_dim):
        if self.indices is None:
            std_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            std_all[self.std != 0] = self.std[self.std != 0]
            return self.mean, std_all
        else:
            mean_all = self.xp.zeros(input_dim, dtype=self.xp.float32)
            mean_all[self.indices] = self.mean
            std_all = self.xp.ones(input_dim, dtype=self.xp.float32)
            non_zero_indices = self.indices[self.std != 0]
            std_all[non_zero_indices] = self.std[self.std != 0]
            return mean_all, std_all

    def transform(self, x):
        if self.mean is None:
            raise AttributeError('[Error] mean is None, call fit beforehand!')
        x = format_x(x)
        mean_all, std_all = self._compute_mean_std_all(x.shape[1])
        return (x - mean_all[None, :]) / std_all[None, :]

    def inverse_transform(self, x):
        if self.mean is None:
            raise AttributeError('[Error] mean is None, call fit beforehand!')
        x = format_x(x)
        mean_all, std_all = self._compute_mean_std_all(x.shape[1])
        return x * std_all[None, :] + mean_all[None, :]
