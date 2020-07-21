import chainer
from chainer import functions


class GeneralReadout(chainer.Link):
    """General submodule for readout part.

    This class can be used for `rsgcn` and `weavenet`.
    Note that this class has no learnable parameter,
    even though this is subclass of `chainer.Link`.
    This class is under `links` module for consistency
    with other readout module.

    Args:
        mode (str):
        activation (callable): activation function
    """

    def __init__(self, mode='sum', activation=None, **kwargs):
        super(GeneralReadout, self).__init__()
        self.mode = mode
        self.activation = activation

    def __call__(self, h, axis=1, **kwargs):
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h

        if self.mode == 'sum':
            y = functions.sum(h, axis=axis)
        elif self.mode == 'max':
            y = functions.max(h, axis=axis)
        elif self.mode == 'summax':
            h_sum = functions.sum(h, axis=axis)
            h_max = functions.max(h, axis=axis)
            y = functions.concat((h_sum, h_max), axis=axis)
        else:
            raise ValueError('mode {} is not supported'.format(self.mode))
        return y


class ScatterGeneralReadout(chainer.Link):
    """General submodule for readout part by scatter operation.

    This class is used in sparse pattern.

    Args:
        mode (str):
        activation (callable): activation function
    """

    def __init__(self, mode='sum', activation=None, **kwargs):
        super(ScatterGeneralReadout, self).__init__()
        self.mode = mode
        self.activation = activation

    def __call__(self, h, batch, **kwargs):
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h

        if self.mode == 'sum':
            y = self.xp.zeros((batch[-1] + 1, h.shape[1]),
                              dtype=self.xp.float32)
            y = functions.scatter_add(y, batch, h)
        else:
            raise ValueError('mode {} is not supported'.format(self.mode))
        return y
