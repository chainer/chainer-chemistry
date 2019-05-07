import chainer


def to_array(x):
    """Convert x into numpy.ndarray or cupy.ndarray"""
    if isinstance(x, chainer.Variable):
        x = x.data
    return x


class BaseScaler(chainer.Link):
    """Base class for scaler.

    x maybe array or Variable
    """

    def fit(self, x, **kwargs):
        """fit parameter from given input `x`.

        It should return self after fitting parameters.
        """
        raise NotImplementedError

    def transform(self, x, **kwargs):
        """transform input `x` using fitted parameters.

        This method should be called after `fit` is called.
        """
        raise NotImplementedError

    def inverse_transform(self, x, **kwargs):
        """inverse operation of `transform`.

        This method should be called after `fit` is called.
        """
        raise NotImplementedError

    def fit_transform(self, x, **kwargs):
        return self.fit(x, **kwargs).transform(x)

    # `__call__` method invokes `forward` method.
    def forward(self, x, **kwargs):
        return self.transform(x, **kwargs)
