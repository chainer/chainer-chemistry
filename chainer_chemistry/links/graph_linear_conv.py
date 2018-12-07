import chainer
from chainer import links
from chainer import functions


class GraphLinear(chainer.Chain):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`chainer.links.Linear`
    """
    def __init__(self, in_channels, out_channels, nobias=False, stride=1,
                 initialW=None, initial_bias=None, pad=0, **kwargs):
        super(GraphLinear, self).__init__()
        self.out_channels = out_channels
        with self.init_scope():
            self.conv = links.Convolution2D(
                in_channels, out_channels, ksize=1, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias,
                **kwargs)

    def __call__(self, x):
        """Forward propagation.

        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray`\
            or :class:`cupy.ndarray`):
                Input array that should be a float array whose ``ndim`` is 3.

                It represents a minibatch of atoms, each of which consists
                of a sequence of molecules. Each molecule is represented
                by integer IDs. The first axis is an index of atoms
                (i.e. minibatch dimension) and the second one an index
                of molecules.

        Returns:
            :class:`chainer.Variable`:
                A 3-dimeisional array.

        """
        h = x
        # (minibatch, atom, ch)
        s0, s1, s2 = h.shape
        # (minibatch, ch, atom)
        h = functions.transpose(h, (0, 2, 1))
        # (minibatch, ch, atom, 1)
        h = functions.reshape(h, (s0, s2, s1, 1))
        # (minibatch, out_ch, atom, 1)
        h = self.conv(h)
        # (minibatch, atom, out_ch, 1)
        h = functions.transpose(h, (0, 2, 1, 3))
        # (minibatch, atom, out_ch)
        h = functions.reshape(h, (s0, s1, self.out_channels))
        return h


if __name__ == '__main__':
    import numpy as np
    bs = 5
    ch = 4
    out_ch = 7
    atom = 3
    x = np.random.rand(bs, atom, ch).astype(np.float32)
    gl = GraphLinear(ch, out_ch)
    y = gl(x)
    print('x', x.shape, 'y', y.shape)  # x (5, 3, 4) y (5, 3, 7)
