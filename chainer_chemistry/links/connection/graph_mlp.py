import numpy

import chainer
from chainer.functions import relu

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GraphMLP(chainer.Chain):

    """Graph MLP layer

    Args:
        channels (list or numpy.ndarray): list of int, representing each
            layer's hidden dim. e.g., if [32, 16], it will construct 2-layer
            MLP with hidden dim 32 and output dim 16.
        in_channels (int or None): input channel size.
        activation (chainer.functions): activation function
    """

    def __init__(self, channels, in_channels=None, activation=relu):
        super(GraphMLP, self).__init__()
        if not isinstance(channels, (list, numpy.ndarray)):
            raise TypeError('channels {} is expected to be list, actual {}'
                            .format(channels, type(channels)))

        channels_list = [in_channels] + list(channels)
        layers = [GraphLinear(channels_list[i], channels_list[i+1])
                  for i in range(len(channels_list) - 1)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
        self.activation = activation

    def __call__(self, x):
        h = x
        for l in self.layers[:-1]:
            h = self.activation(l(h))
        h = self.layers[-1](h)
        return h
