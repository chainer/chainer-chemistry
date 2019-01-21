import chainer
from chainer import functions


class GeneralReadout(chainer.Link):
    """General submodule for readout part.

    This class can be used for rsgcn and weavenet.

    Args:
        mode (str):
        activation (callable): activation function
    """

    def __init__(self, mode='sum', activation=None):
        super(GeneralReadout, self).__init__()
        self.mode = mode
        self.activation = activation

    def __call__(self, x, axis=1):
        if self.activation is not None:
            h = self.activation(x)
        else:
            h = x

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
