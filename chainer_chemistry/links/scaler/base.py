import chainer


def to_array(x):
    """Convert x into numpy.ndarray or cupy.ndarray"""
    if isinstance(x, chainer.Variable):
        x = x.data
    return x


class BaseScaler(chainer.Link):
    # x maybe array or Variable

    def fit(self, x, **kwargs):
        raise NotImplementedError

    def transform(self, x, **kwargs):
        raise NotImplementedError

    def inverse_transform(self, x, **kwargs):
        raise NotImplementedError

    def fit_transform(self, x, **kwargs):
        return self.fit(x, **kwargs).transform(x)

    # def serialize(self):
    # Already implemented as `Link`
    #     raise NotImplementedError
