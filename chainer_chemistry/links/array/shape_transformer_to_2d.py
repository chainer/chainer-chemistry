import chainer
from chainer import functions


class ShapeTransformerTo2D(chainer.Link):
    """Transforms input array `x` to 2-dim and reverts.

    It converts array to be 2-dim, where 1th axis is `axis` and the rest is
    gathered to 0th axis.

    Note that this class does not have any parameters
    but behaves as "function wrapper" which has internal attribute to
    `transform` and `inverse_transform`.

    Args:
        axis (int): feature axis, which will be 1st axis.
    """

    def __init__(self, axis=1):
        super(ShapeTransformerTo2D, self).__init__()
        self.original_shape = None
        self.transpose_order = None
        self.axis = axis

    def transform(self, x):
        self.original_shape = x.shape
        axis = self.axis
        if axis < 0:
            axis += x.ndim

        transpose_order = [i for i in range(x.ndim)]
        transpose_order.pop(axis)
        transpose_order = transpose_order + [axis]
        x = functions.transpose(x, tuple(transpose_order))
        x = functions.reshape(x, (-1, self.original_shape[axis]))
        self.transpose_order = transpose_order
        return x

    def inverse_transform(self, x):
        if x.ndim != 2:
            raise ValueError(
                "[ERROR] Unexpected value x.shape={}, 2-dim array is expected"
                .format(x.shape))
        if self.original_shape is None:
            raise AttributeError(
                '[Error] original_shape is None, call transform beforehand!')

        ndim = len(self.original_shape)
        axis = self.axis
        if axis < 0:
            axis += ndim
        inverse_transpose_order = [i for i in range(ndim - 1)]
        inverse_transpose_order.insert(axis, ndim-1)
        x = functions.reshape(x, tuple([self.original_shape[i]
                                        for i in self.transpose_order]))
        x = functions.transpose(x, tuple(inverse_transpose_order))
        return x
